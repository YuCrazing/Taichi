import taichi as ti
import numpy as np
import time



ti.init(arch=ti.gpu)


n = 400
dt = 0.05
dx = 1/n

rho = 10

# 1, 2, 3
RK = 3

#
enable_BFECC = True

#
enable_clipping = False

colors = ti.Vector(3, dt=ti.f32, shape=(n, n))
new_colors = ti.Vector(3, dt=ti.f32, shape=(n, n))
new_new_colors = ti.Vector(3, dt=ti.f32, shape=(n, n))

velocities = ti.Vector(2, dt=ti.f32, shape=(n, n))
new_velocities = ti.Vector(2, dt=ti.f32, shape=(n, n))
new_new_velocities = ti.Vector(2, dt=ti.f32, shape=(n, n))

pressures = ti.var(dt=ti.f32, shape=(n, n))
new_pressures = ti.var(dt=ti.f32, shape=(n, n))


# screen center. The simualation area is (0, 0) to (1, 1)
center = ti.Vector([0.5, 0.5])

# cell center
stagger = ti.Vector([0.5, 0.5])


@ti.func
def I(i, j):
	return ti.Vector([i, j])


@ti.func
def vel(p):
	# rotation
	# return ti.Vector([p.y-center.y, center.x-p.x])
	return sample_bilinear(velocities, p)


@ti.kernel
def init_color_field():
	# super sampling
	# for i, j in ti.ndrange(4*n, 4*n):
	# 	d = ti.Vector([i/4*dx, j/4*dx]) - center
	# 	if d.norm_sqr() < 0.1:
	# 		x = i/4*dx
	# 		if x < 1/3:
	# 			colors[i//4, j//4] += ti.Vector([0.0, 0.0, 0.0])/16
	# 		elif x < 2/3:
	# 			colors[i//4, j//4] += ti.Vector([1.0, 1.0, 1.0])/16
	# 		else:
	# 			colors[i//4, j//4] += ti.Vector([1.0, 0.0, 0.0])/16
	# 	else:
	# 		colors[i//4, j//4] += ti.Vector([1.0, 1.0, 1.0])/16

	# random
	for i in ti.grouped(colors):
		colors[i] = ti.Vector([ti.random(), ti.random(), ti.random()])


@ti.kernel
def init_velocity_field():
	# rotation
	for i in ti.grouped(velocities):
		p = (i + stagger) * dx
		d = p - center
		# if d.norm_sqr() < 0.2:
			# velocities[i] = ti.Vector([p.y-center.y, center.x-p.x])
		velocities[i] = ti.Vector([0.0, 0.0])
		


@ti.func
def clamp(p):
	# clamp p to [0.5*dx, 1-dx+0.5*dx), i.e. clamp cell index to [0, n-1)
	for d in ti.static(range(p.n)):
		p[d] = min(1 - dx + stagger[d]*dx - 1e-4, max(p[d], stagger[d]*dx))
	return p

@ti.func
def sample_bilinear(field, p):

	p = clamp(p)

	grid_f = p * n - stagger
	grid_i = ti.cast(ti.floor(grid_f), ti.i32)

	d = grid_f - grid_i
	
	return field[ grid_i ] * (1-d.x)*(1-d.y) + field[ grid_i+I(1, 0) ] * d.x*(1-d.y) + field[ grid_i+I(0, 1) ] * (1-d.x)*d.y + field[ grid_i+I(1, 1) ] * d.x*d.y

@ti.func
def sample_min(field, p):

	p = clamp(p)

	grid_f = p * n - stagger
	grid_i = ti.cast(ti.floor(grid_f), ti.i32)
	

	color = ti.Vector([0.0, 0.0, 0.0])

	for d in ti.static(range(color.n)):
		color[d] = min( field[ grid_i ][d], field[ grid_i+I(1, 0) ][d], field[ grid_i+I(0, 1) ][d], field[ grid_i+I(1, 1) ][d] )

	return color

@ti.func
def sample_max(field, p):

	p = clamp(p)

	grid_f = p * n - stagger
	grid_i = ti.cast(ti.floor(grid_f), ti.i32)
	
	color = ti.Vector([0.0, 0.0, 0.0])

	for d in ti.static(range(color.n)):
		color[d] = max( field[ grid_i ][d], field[ grid_i+I(1, 0) ][d], field[ grid_i+I(0, 1) ][d], field[ grid_i+I(1, 1) ][d] )

	return color

@ti.func
def backtrace(p, dt):

	if ti.static(RK == 1):
		return p - vel(p) * dt
	elif ti.static(RK == 2):
		p_mid = p - vel(p) * dt * 0.5
		return p - vel(p_mid) * dt
	elif ti.static(RK == 3):
		v_p = vel(p)
		p_mid = p - v_p * dt * 0.5
		v_mid = vel(p_mid)
		p_mid_mid = p - v_mid * dt * 0.75
		v_mid_mid = vel(p_mid_mid)
		return p - (2/9 * v_p + 1/3*v_mid + 4/9*v_mid_mid) * dt



@ti.func
def semi_lagrangian(field, new_field, dt):
	for i in ti.grouped(field):
		p = (i + stagger) * dx
		new_field[i] = sample_bilinear(field, backtrace(p, dt))



@ti.func
def BFECC(field, new_field, new_new_field, dt):
	
	semi_lagrangian(field, new_field, dt)
	semi_lagrangian(new_field, new_new_field, -dt)

	for i in ti.grouped(field):
		
		new_field[i] = new_field[i] - 0.5 * (new_new_field[i] - field[i])

		if ti.static(enable_clipping):
			
			source_pos = backtrace( (i + stagger) * dx, dt )
			mi = sample_min(field, source_pos)
			mx = sample_max(field, source_pos)

			if new_field[i].x < mi.x or new_field[i].y < mi.y or new_field[i].z < mi.z or new_field[i].x > mx.x or new_field[i].y > mx.y or new_field[i].z > mx.z:
				new_field[i] = sample_bilinear(field, source_pos)
				
				# runtime error
				# break

@ti.kernel
def advect(field:ti.template(), new_field:ti.template(), new_new_field:ti.template(), dt:ti.f32):

	if ti.static(enable_BFECC):
		BFECC(field, new_field, new_new_field, dt)
	else:
		semi_lagrangian(field, new_field, dt)
		
	for i in ti.grouped(field):
		field[i] = new_field[i]


@ti.kernel
def jacobi():
	for i, j in velocities:
		# c = ti.Vector([i + stagger.x, j + stagger.y]) * dx
		c = ti.Vector([i + stagger.x, j + stagger.y]) * dx
		# l = c - ti.Vector([1, 0])
		# r = c + ti.Vector([1, 0])
		# d = c - ti.Vector([0, 1])
		# u = c + ti.Vector([0, 1])		
		l = c - ti.Vector([stagger.x, 0])
		r = c + ti.Vector([stagger.x, 0])
		d = c - ti.Vector([0, stagger.y])
		u = c + ti.Vector([0, stagger.y])

		v_c = sample_bilinear(velocities, c)
		v_l = sample_bilinear(velocities, l).x
		v_r = sample_bilinear(velocities, r).x
		v_d = sample_bilinear(velocities, d).y
		v_u = sample_bilinear(velocities, u).y

		# p_c = sample_bilinear(pressures, c)
		p_l = sample_bilinear(pressures, l)
		p_r = sample_bilinear(pressures, r)
		p_d = sample_bilinear(pressures, d)
		p_u = sample_bilinear(pressures, u)

		k = 4
		if i == 0: 
			v_l = -v_c.x
			p_l = 0.0
			k -= 1
		if i == n-1:
			v_r = -v_c.x
			p_r = 0.0
			k -= 1
		if j == 0:
			v_d = -v_c.y
			p_d = 0.0
			k -= 1
		if j == n-1:
			v_u = -v_c.y
			p_u = 0.0
			k -= 1

		div = (v_r - v_l + v_u - v_d) * rho / dt * dx * 0.25

		pressures[i, j] = (p_l + p_r + p_d + p_u - div) / k


def solve_pressure(num_iter):
	for i in range(num_iter):
		jacobi()


@ti.kernel
def projection():
	for i, j in velocities:
		c = ti.Vector([i + stagger.x, j + stagger.y]) * dx
		l = c - ti.Vector([stagger.x, 0])
		r = c + ti.Vector([stagger.x, 0])
		d = c - ti.Vector([0, stagger.y])
		u = c + ti.Vector([0, stagger.y])
		p_l = sample_bilinear(pressures, l)
		p_r = sample_bilinear(pressures, r)
		p_d = sample_bilinear(pressures, d)
		p_u = sample_bilinear(pressures, u)

		velocities[i, j] -=  ti.Vector([p_r - p_l, p_u - p_d]) / dx * dt / rho

@ti.kernel
def apply_force(pre_mouse_pos:ti.ext_arr(), cur_mouse_pos:ti.ext_arr()):

	p = ti.Vector([cur_mouse_pos[0], cur_mouse_pos[1]])
	pre_p = ti.Vector([pre_mouse_pos[0], pre_mouse_pos[1]])

	dp = p - pre_p
	dp = dp / max(1e-5, dp.norm())

	for i, j in velocities:


		d2 = (ti.Vector([(i+stagger.x)*dx, (j+stagger.y)*dx]) - p).norm_sqr()

		velocities[i, j] += dp * dt * ti.exp(-d2/0.01)



init_color_field()
init_velocity_field()


gui = ti.GUI("Fluid 2D", (n, n))


# result_dir = "./fluid_2d"
# video_manager = ti.VideoManager(output_dir=result_dir, framerate=30, automatic_build=False)

pre_mouse_pos = None
cur_mouse_pos = None


for frame in range(30000):
# while gui.running:	
	advect(colors, new_colors, new_new_colors, dt)
	advect(velocities, new_velocities, new_new_velocities, dt)
	# time.sleep(1)


	gui.get_event(ti.GUI.PRESS)

	if gui.is_pressed(ti.GUI.LMB):
		pre_mouse_pos = cur_mouse_pos
		cur_mouse_pos = np.array(gui.get_cursor_pos(), dtype=np.float32)
		if pre_mouse_pos is None:
			pre_mouse_pos = cur_mouse_pos
	else:
		pre_mouse_pos = cur_mouse_pos = None

	if pre_mouse_pos is not None: 
		print(pre_mouse_pos, cur_mouse_pos)
		apply_force(pre_mouse_pos, cur_mouse_pos)



	solve_pressure(200)

	projection()

	gui.set_image(colors)

	gui.text(content=f'RK {RK}', pos=(0, 0.98), color=0x0)
	if enable_BFECC: 
		gui.text(content=f'BFECC', pos=(0, 0.94), color=0x0)
		if enable_clipping: 
			gui.text(content=f'Clipped', pos=(0, 0.90), color=0x0)
	
	gui.show()

	# video_manager.write_frame(colors.to_numpy())

# video_manager.make_video(gif=True, mp4=True)