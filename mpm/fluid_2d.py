import taichi as ti
import numpy as np
import time



ti.init(arch=ti.gpu)


res = 512
dt = 2.0e-3

# 1, 2, 3
RK = 3

#
enable_BFECC = True

#
enable_clipping = True


rho = 1
jacobi_iters = 300


m_g = 64
m_p = 128
n_grid = m_g*m_g
n_particle = m_p*m_p

dx = 1/m_g


eps = 1e-5

debug = False



# colors = ti.Vector(3, dt=ti.f32, shape=(m_g, m_g))
# new_colors = ti.Vector(3, dt=ti.f32, shape=(n, n))
# new_new_colors = ti.Vector(3, dt=ti.f32, shape=(n, n))

velocities = ti.Vector(2, dt=ti.f32, shape=(m_g, m_g))
# new_velocities = ti.Vector(2, dt=ti.f32, shape=(m_g, m_g))
# new_new_velocities = ti.Vector(2, dt=ti.f32, shape=(n, n))

pressures = ti.var(dt=ti.f32, shape=(m_g, m_g))
new_pressures = ti.var(dt=ti.f32, shape=(m_g, m_g))

divergences = ti.var(dt=ti.f32, shape=(m_g, m_g))



weights = ti.var(dt=ti.f32, shape=(m_g, m_g))

# 0: fluid
# 1: air
# 2: solid
FLUID = 0
AIR = 1
SOLID = 2
types = ti.var(dt=ti.i32, shape=(m_g, m_g))

particle_velocity = ti.Vector(2, dt=ti.f32, shape=n_particle)
particle_position = ti.Vector(2, dt=ti.f32, shape=n_particle)




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



# @ti.kernel
# def init_velocity_field():
# 	# rotation
# 	for i in ti.grouped(velocities):
# 		p = (i + stagger) * dx
# 		d = p - center
# 		# if d.norm_sqr() < 0.2:
# 			# velocities[i] = ti.Vector([p.y-center.y, center.x-p.x])
# 		velocities[i] = ti.Vector([0.0, 0.0])
		


@ti.func
def clamp(p):
	# clamp p to [0.5*dx, 1-dx+0.5*dx), i.e. clamp cell index to [0, n-1)
	for d in ti.static(range(p.n)):
		p[d] = min(1 - dx + stagger[d]*dx - 1e-4, max(p[d], stagger[d]*dx))
	return p

@ti.func
def sample_bilinear(field, p):

	p = clamp(p)

	grid_f = p * m_g - stagger
	grid_i = ti.cast(ti.floor(grid_f), ti.i32)

	d = grid_f - grid_i
	
	return field[ grid_i ] * (1-d.x)*(1-d.y) + field[ grid_i+I(1, 0) ] * d.x*(1-d.y) + field[ grid_i+I(0, 1) ] * (1-d.x)*d.y + field[ grid_i+I(1, 1) ] * d.x*d.y


@ti.kernel
def init_grid():
	for i, j in types:
		if i <= 1 or i >= m_g-2 or j <=1 or j >= m_g-2:
			types[i, j] = SOLID


@ti.kernel
def init_particle():
	for i in particle_position:
		particle_position[i] = ti.Vector([i%m_p / m_p / 2, i//m_p / m_p / 2]) + ti.Vector([0.1, 0.25])
		# particle_position[i] = ti.Vecxtor([ti.random(), ti.random()])


# @ti.kernel
# def advect_particle():
# 	for i in particle_velocity:
# 		particle_velocity[i] = particle_velocity[i] + ti.Vector([0, -9.8]) * dt * 0.01 

# 	# print(particle_position[0])


@ti.kernel
def particle_to_grid():

	# clear grid
	for k in ti.grouped(velocities):
		velocities[k] = ti.Vector([0.0, 0.0])
		weights[k] = 0.0
		pressures[k] = 0.0

		if types[k] != SOLID:
			types[k] = AIR
			# velocities[k] += ti.Vector([0.0, -9.8]) * dt

	for k in particle_velocity:
		p = particle_position[k]
		p_g = p * m_g

		# change cell type to 'fluid'
		types[p_g.cast(int)] = FLUID


		# find left bottom corner
		base = (p_g - stagger).cast(int)
		fx = p_g - base.cast(float)
		# quadratic B-spline
		w = [0.5 * (1.5-fx)**2, 0.75 - (fx-1)**2, 0.5 * (fx-0.5)**2]

		for i in ti.static(range(3)):
			for j in ti.static(range(3)):
				offset = ti.Vector([i, j])
				weight = w[i][0] * w[j][1]
				velocities[base + offset] += weight * particle_velocity[k]
				weights[base + offset] += weight 

	for k in ti.grouped(weights):
		weight = weights[k]
		if weight > 0:
			velocities[k] = velocities[k] / weight
			# print(k.x, k.y, velocities[k])

	for k in ti.grouped(velocities):
		if types[k] != SOLID:
			velocities[k] += ti.Vector([0.0, -9.8]) * dt


@ti.kernel
def solve_divergence():

	for i in ti.grouped(divergences):
		if types[i] != FLUID:
			divergences[i] = 0.0
			pressures[i] = 0.0

	for i, j in velocities:
		if types[i, j] == FLUID:
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


			div = v_r - v_l + v_u - v_d

			if types[i-1, j] == SOLID: 
				v_l = -v_c.x
				# div += v_l
			if types[i+1, j] == SOLID: 
				v_r = -v_c.x
				# div -= v_r
			if types[i, j-1] == SOLID: 
				v_d = -v_c.y
				# div += v_d
			if types[i, j+1] == SOLID: 
				v_u = -v_c.y
				# div -= v_u

			div = v_r - v_l + v_u - v_d

			divergences[i, j] = div / (2*dx)


@ti.kernel
def pressure_jacobi(pressures:ti.template(), new_pressures:ti.template()):

	for i, j in velocities:
		if types[i, j] == FLUID:
			c = ti.Vector([i + stagger.x, j + stagger.y]) * dx
			l = c - ti.Vector([1, 0]) * dx
			r = c + ti.Vector([1, 0]) * dx
			d = c - ti.Vector([0, 1]) * dx
			u = c + ti.Vector([0, 1]) * dx

			p_l = sample_bilinear(pressures, l)
			p_r = sample_bilinear(pressures, r)
			p_d = sample_bilinear(pressures, d)
			p_u = sample_bilinear(pressures, u)

			k = 4
			if types[i-1, j] == SOLID:
				p_l = 0.0
				k -= 1
			if types[i+1, j] == SOLID:
				p_r = 0.0
				k -= 1
			if types[i, j-1] == SOLID:
				p_d = 0.0
				k -= 1
			if types[i, j+1] == SOLID:
				p_u = 0.0
				k -= 1

			if types[i-1, j] == AIR:
				p_l = 0.0
			if types[i+1, j] == AIR:
				p_r = 0.0
			if types[i, j-1] == AIR:
				p_d = 0.0
			if types[i, j+1] == AIR:
				p_u = 0.0

			new_pressures[i, j] = 1/3 * pressures[i, j] + 2/3 *  ( p_l + p_r + p_d + p_u - divergences[i, j] * rho / dt * (dx*dx) ) / k



@ti.kernel
def projection():
	for i, j in velocities:
		if types[i, j] == FLUID:
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


			new_v = velocities[i, j] - grad_p / rho * dt

			if types[i-1, j] == SOLID:
				new_v.x = 0
			if types[i+1, j] == SOLID:
				new_v.x = 0
			if types[i, j-1] == SOLID:
				new_v.y = 0
			if types[i, j+1] == SOLID:
				new_v.y = 0

			velocities[i, j] = new_v

			# if i == 5 and j == 5: 
			# 	print(pressures[i, j])
	
	# print(pressures[5, 5])
# @ti.func
# def clamp_pos(p):
#     return ti.Vector([max(min(0.95, p[0]), 0.05), max(min(0.95, p[1]), 0.05)])

@ti.kernel
def grid_to_particle():


	for k in particle_velocity:
		p = particle_position[k]
		p_g = p * m_g

		# find left bottom corner
		base = (p_g - stagger).cast(int)
		fx = p_g - base.cast(float)
		# quadratic B-spline
		w = [0.5 * (1.5-fx)**2, 0.75 - (fx-1)**2, 0.5 * (fx-0.5)**2]

		new_v = ti.Vector.zero(ti.f32, 2)

		for i in ti.static(range(3)):
			for j in ti.static(range(3)):
				offset = ti.Vector([i, j])
				weight = w[i][0] * w[j][1]
				new_v += weight * velocities[base + offset]


		new_p = p + new_v * dt


		# damp = 0.99

		# normal = ti.Vector([0.0, 0.0])

		# if new_p.x < dx:
		# 	new_p.x = dx
		# 	normal.x += -1.0
		# if new_p.x > 1 - dx:
		# 	new_p.x = 1 - dx
		# 	normal.x += 1.0
		# if new_p.y < 0.05:
		# 	new_p.y = 0.05
		# 	normal.y += -1.0
		# if new_p.y > 0.95:
		# 	new_p.y = 0.95
		# 	normal.y += 1.0

		# nl = normal.norm()
		# vl = new_v.norm()

		# if nl > 0.1 and vl > 0.1:
		# 	normal /= nl
		# 	new_v -= normal * vl


		if new_p.x < dx*2:
			new_p.x = dx*2
			new_v.x = 0
		if new_p.x >= 1 - dx*2:
			new_p.x = 1 - dx*2 - eps
			new_v.x = 0
		if new_p.y < dx*2:
			new_p.y = dx*2
			new_v.y = 0
		if new_p.y >= 1 - dx*2:
			new_p.y = 1 - dx*2 - eps
			new_v.y = 0

		# ???????????????
		particle_position[k] = new_p

		particle_velocity[k] = new_v




def step():
	# advect_particle()

	particle_to_grid()

	solve_divergence()
	
	for i in range(jacobi_iters):
		global pressures
		global new_pressures
		pressure_jacobi(pressures, new_pressures)
		# global new_pressures
		pressures, new_pressures = new_pressures, pressures


	projection()

	grid_to_particle()





@ti.kernel
def test():
	print(particle_velocity[0])


# init_velocity_field()


init_grid()
init_particle()




gui = ti.GUI("Fluid 2D", (res, res))


# result_dir = "./fluid_2d"
# video_manager = ti.VideoManager(output_dir=result_dir, framerate=30, automatic_build=False)

# pre_mouse_pos = None
# cur_mouse_pos = None


for frame in range(450000):

	# for i in range(10):
	# if frame <= 42: 
	# 	step()
	step()

	# if frame <= 42:
		# test()

	# time.sleep(1)

	# break
	if debug:
		for i in range(m_g):
			for j in range(m_g):
				color = 0
				if types[i, j] == FLUID:
					color = 0xFFFFFF
				elif types[i, j] == AIR:
					color = 0x0000FF
				elif types[i, j] == SOLID:
					color = 0xFF0000
				gui.circle([(i+0.5)*dx, (j+0.5)*dx], radius = 3, color = color)
				# gui.line([i*dx, j*dx], [i*dx, (j+1)*dx], color = 0xFF0000)
				# gui.line([i*dx, (j+1)*dx], [(i+1)*dx, (j+1)*dx], color = 0xFF0000)
				# gui.line([(i+1)*dx, j*dx], [(i+1)*dx, (j+1)*dx], color = 0xFF0000)
				# gui.line([(i+1)*dx, j*dx], [i*dx, j*dx], color = 0xFF0000)

	gui.circles(particle_position.to_numpy(), radius=1, color=0xFFFFFF)


	
	gui.show()

	# video_manager.write_frame(colors.to_numpy())

# video_manager.make_video(gif=True, mp4=True)