import taichi as ti
import numpy as np
import time



ti.init(arch=ti.gpu)


n = 512
dt = 0.03
dx = 1/n

# 1, 2, 3
RK = 3

#
enable_BFECC = True

#
enable_clipping = True


rho = 1
jacobi_iters = 160



colors = ti.Vector(3, dt=ti.f32, shape=(n, n))
new_colors = ti.Vector(3, dt=ti.f32, shape=(n, n))
new_new_colors = ti.Vector(3, dt=ti.f32, shape=(n, n))

velocities = ti.Vector(2, dt=ti.f32, shape=(n, n))
new_velocities = ti.Vector(2, dt=ti.f32, shape=(n, n))
new_new_velocities = ti.Vector(2, dt=ti.f32, shape=(n, n))

pressures = ti.var(dt=ti.f32, shape=(n, n))
new_pressures = ti.var(dt=ti.f32, shape=(n, n))

divergences = ti.var(dt=ti.f32, shape=(n, n))


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
	# random
	for i in ti.grouped(colors):
		# colors[i] = ti.Vector([ti.random(), ti.random(), ti.random()])
		colors[i] = ti.Vector([0.0, 0.0, 0.0])
		# colors[i] = ti.Vector([1.0, 1.0, 1.0])


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
	
	return min( field[ grid_i ], field[ grid_i+I(1, 0) ], field[ grid_i+I(0, 1) ], field[ grid_i+I(1, 1) ] )


@ti.func
def sample_max(field, p):

	p = clamp(p)

	grid_f = p * n - stagger
	grid_i = ti.cast(ti.floor(grid_f), ti.i32)
	
	return max( field[ grid_i ], field[ grid_i+I(1, 0) ], field[ grid_i+I(0, 1) ], field[ grid_i+I(1, 1) ] )


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

			for d in ti.static(range(mi.n)):
				if new_field[i][d] < mi[d] or new_field[i][d] > mx[d]:
					new_field[i] = sample_bilinear(field, source_pos)


@ti.kernel
def advect(field:ti.template(), new_field:ti.template(), new_new_field:ti.template(), dt:ti.f32):

	if ti.static(enable_BFECC):
		BFECC(field, new_field, new_new_field, dt)
	else:
		semi_lagrangian(field, new_field, dt)



@ti.kernel
def solve_divergence():
	for i, j in velocities:
		c = ti.Vector([i + stagger.x, j + stagger.y]) * dx
		l = c - ti.Vector([1, 0]) * dx
		r = c + ti.Vector([1, 0]) * dx
		d = c - ti.Vector([0, 1]) * dx
		u = c + ti.Vector([0, 1]) * dx
		v_c = sample_bilinear(velocities, c)
		v_l = sample_bilinear(velocities, l).x
		v_r = sample_bilinear(velocities, r).x
		v_d = sample_bilinear(velocities, d).y
		v_u = sample_bilinear(velocities, u).y


		if i == 0: 
			v_l = -v_c.x
		if i == n-1:
			v_r = -v_c.x
		if j == 0:
			v_d = -v_c.y
		if j == n-1:
			v_u = -v_c.y

		divergences[i, j] = (v_r - v_l + v_u - v_d) / (2*dx)


@ti.kernel
def pressure_jacobi(pressures:ti.template(), new_pressures:ti.template()):

	for i, j in pressures:
		c = ti.Vector([i + stagger.x, j + stagger.y]) * dx
		l = c - ti.Vector([1, 0]) * dx
		r = c + ti.Vector([1, 0]) * dx
		d = c - ti.Vector([0, 1]) * dx
		u = c + ti.Vector([0, 1]) * dx

		p_l = sample_bilinear(pressures, l)
		p_r = sample_bilinear(pressures, r)
		p_d = sample_bilinear(pressures, d)
		p_u = sample_bilinear(pressures, u)

		new_pressures[i, j] = ( p_l + p_r + p_d + p_u - divergences[i, j] * rho / dt * (dx*dx) ) * 0.25



@ti.kernel
def projection():
	for i, j in velocities:
		c = ti.Vector([i + stagger.x, j + stagger.y]) * dx
		l = c - ti.Vector([1, 0]) * dx
		r = c + ti.Vector([1, 0]) * dx
		d = c - ti.Vector([0, 1]) * dx
		u = c + ti.Vector([0, 1]) * dx
		p_c = sample_bilinear(pressures, c)
		p_l = sample_bilinear(pressures, l)
		p_r = sample_bilinear(pressures, r)
		p_d = sample_bilinear(pressures, d)
		p_u = sample_bilinear(pressures, u)

		grad_p = ti.Vector([p_r - p_l, p_u - p_d]) / (2*dx)

		# if i == 0:
		# 	grad_p.x = (p_r - p_c) / (dx/2)
		# if i == n-1:
		# 	grad_p.x = (p_c - p_l) / (dx/2)
		# if j == 0:
		# 	grad_p.y = (p_u - p_c) / (dx/2)
		# if j == n-1:
		# 	grad_p.y = (p_c - p_d) / (dx/2)

		velocities[i, j] = velocities[i, j] - grad_p / rho * dt


@ti.kernel
def apply_force(pre_mouse_pos:ti.ext_arr(), cur_mouse_pos:ti.ext_arr()):

	p = ti.Vector([cur_mouse_pos[0], cur_mouse_pos[1]])
	pre_p = ti.Vector([pre_mouse_pos[0], pre_mouse_pos[1]])

	dp = p - pre_p
	dp = dp / max(1e-5, dp.norm())

	color = (ti.Vector([ti.random(), ti.random(), ti.random()]) * 0.7 + ti.Vector([0.1, 0.1, 0.1]) * 0.3)

	for i, j in velocities:


		d2 = (ti.Vector([(i+stagger.x)*dx, (j+stagger.y)*dx]) - p).norm_sqr()

		radius = 0.0001
		velocities[i, j] = velocities[i, j] + dp * dt * ti.exp(-d2/radius) * 20


		if dp.norm() > 0.5:
			colors[i, j] = colors[i, j] + ti.exp(-d2 * (4 / (1 / 15)**2)) * color



@ti.kernel
def decay_color():
	for i in ti.grouped(colors):
		colors[i] = colors[i] * 0.99


init_color_field()
init_velocity_field()


gui = ti.GUI("Fluid 2D", (n, n))


result_dir = "./result"
video_manager = ti.VideoManager(output_dir=result_dir, framerate=30, automatic_build=False)

pre_mouse_pos = None
cur_mouse_pos = None


for frame in range(450):

	advect(velocities, new_velocities, new_new_velocities, dt)
	advect(colors, new_colors, new_new_colors, dt)
	velocities, new_velocities = new_velocities, velocities
	colors, new_colors = new_colors, colors


	gui.get_event(ti.GUI.PRESS)

	if gui.is_pressed(ti.GUI.LMB):
		pre_mouse_pos = cur_mouse_pos
		cur_mouse_pos = np.array(gui.get_cursor_pos(), dtype=np.float32)
		if pre_mouse_pos is None:
			pre_mouse_pos = cur_mouse_pos
		apply_force(pre_mouse_pos, cur_mouse_pos)
	else:
		pre_mouse_pos = cur_mouse_pos = None


	decay_color()

	solve_divergence()


	for i in range(jacobi_iters):
		pressure_jacobi(pressures, new_pressures)
		pressures, new_pressures = new_pressures, pressures

	projection()



	gui.set_image(colors)

	gui.text(content=f'RK {RK}', pos=(0, 0.98), color=0xFFFFFF)
	if enable_BFECC: 
		gui.text(content=f'BFECC', pos=(0, 0.94), color=0xFFFFFF)
		if enable_clipping: 
			gui.text(content=f'Clipped', pos=(0, 0.90), color=0xFFFFFF)
	
	gui.show()

	# video_manager.write_frame(colors.to_numpy())

# video_manager.make/_video(gif=True, mp4=True)