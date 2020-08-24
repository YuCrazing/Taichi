import taichi as ti
import numpy
import time



ti.init(arch=ti.cpu)


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

@ti.func
def jacobi_():


@ti.func
def solve_pressure():
	for i, j in velocities:
		lap = 
		pressures[i, j] = 

@ti.kernel
def advect(field:ti.template(), new_field:ti.template(), new_new_field:ti.template(), dt:ti.f32):

	if ti.static(enable_BFECC):
		BFECC(field, new_field, new_new_field, dt)
	else:
		semi_lagrangian(field, new_field, dt)
		
	for i in ti.grouped(field):
		field[i] = new_field[i]



init_color_field()
init_velocity_field()


gui = ti.GUI("Fluid 2D", (n, n))


# result_dir = "./fluid_2d"
# video_manager = ti.VideoManager(output_dir=result_dir, framerate=30, automatic_build=False)

for frame in range(300000):
	
	advect(colors, new_colors, new_new_colors, dt)
	advect(velocities, new_velocities, new_new_velocities, dt)
	# time.sleep(1)

	gui.set_image(colors)

	gui.text(content=f'RK {RK}', pos=(0, 0.98), color=0x0)
	if enable_BFECC: 
		gui.text(content=f'BFECC', pos=(0, 0.94), color=0x0)
		if enable_clipping: 
			gui.text(content=f'Clipped', pos=(0, 0.90), color=0x0)
	
	gui.show()

	# video_manager.write_frame(colors.to_numpy())

# video_manager.make_video(gif=True, mp4=True)