import taichi as ti
import numpy as np
import time


# apply force after grid normalization (or explosion)
# pressure in air cells should be 0 (or volume shrink quickly)
# velocities which are next to solid should be 0
# velocity in air cells may not be 0


ti.init(arch=ti.gpu, default_fp=ti.f32)


res = 512
dt = 2e-2 #2e-3 #2e-2
substep = 1


rho = 1000
jacobi_iters = 300
jacobi_damped_para = 0.67
FLIP_blending = 0.3



m_g = 128
n_grid = m_g*m_g
n_particle = n_grid*4

length = 10.0
dx = length/m_g
inv_dx = 1/dx

# solid boundary
boundary_width = 2

eps = 1e-5

# show grid types
debug = False


# MAC grid
velocities_u = ti.var(dt=ti.f32, shape=(m_g+1, m_g))
velocities_v = ti.var(dt=ti.f32, shape=(m_g, m_g+1))

last_velocities_u = ti.var(dt=ti.f32, shape=(m_g+1, m_g))
last_velocities_v = ti.var(dt=ti.f32, shape=(m_g, m_g+1))

weights_u = ti.var(dt=ti.f32, shape=(m_g+1, m_g))
weights_v = ti.var(dt=ti.f32, shape=(m_g, m_g+1))


pressures = ti.var(dt=ti.f32, shape=(m_g, m_g))
new_pressures = ti.var(dt=ti.f32, shape=(m_g, m_g))
divergences = ti.var(dt=ti.f32, shape=(m_g, m_g))

FLUID = 0
AIR = 1
SOLID = 2

types = ti.var(dt=ti.i32, shape=(m_g, m_g))

particle_velocity = ti.Vector(2, dt=ti.f32, shape=n_particle)
particle_position = ti.Vector(2, dt=ti.f32, shape=n_particle)


@ti.kernel
def init_grid():
	for i, j in types:
		if i < boundary_width or i >= m_g-boundary_width or j < boundary_width or j >= m_g-boundary_width:
			types[i, j] = SOLID


# should not generate particles in solid cells
@ti.kernel
def init_particle():
	for i in particle_position:
		# particle_position[i] = ti.Vector([ti.random(), ti.random()]) * 0.6 * 5  + ti.Vector([0.5, 0.5])
		particle_position[i] = ti.Vector([ti.random(), ti.random()]) * 5 + ti.Vector([0.5, 0.5])
		particle_velocity[i] = ti.Vector([0.0, 0.0])


@ti.func
def is_valid(i, j):
	return i >= 0 and i <= m_g-1 and j >= 0 and j <= m_g-1

@ti.func
def is_solid(i, j):
	return is_valid(i, j) and types[i, j] == SOLID

@ti.func
def is_air(i, j):
	return is_valid(i, j) and types[i, j] == AIR

@ti.func
def is_fluid(i, j):
	return is_valid(i, j) and types[i, j] == FLUID


@ti.kernel
def handle_boundary():

	for i, j in velocities_u:
		if is_solid(i-1, j) or is_solid(i, j):
			velocities_u[i, j] = 0.0	
	
	for i, j in velocities_v:
		if is_solid(i, j-1) or is_solid(i, j):
			velocities_v[i, j] = 0.0

@ti.kernel
def init_step():

	for k in ti.grouped(velocities_u):
		velocities_u[k] = 0.0
		weights_u[k] = 0.0
	
	for k in ti.grouped(velocities_v):
		velocities_v[k] = 0.0
		weights_v[k] = 0.0

	for k in ti.grouped(pressures):
		if is_air(k.x, k.y):
			pressures[k] = 0.0
			new_pressures[k] = 0.0


@ti.kernel
def mark_cell():
	for i, j in types:
		if not is_solid(i, j):
			types[i, j] = AIR


	for k in particle_velocity:
		grid = (particle_position[k] * inv_dx).cast(int)
		if not is_solid(grid.x, grid.y):
			types[grid] = FLUID


@ti.func
def scatter(grid_v, grid_m, xp, vp, stagger):
	base = (xp * inv_dx - (stagger + 0.5)).cast(ti.i32)
	fx = xp * inv_dx - (base.cast(ti.f32) + stagger)

	w = [0.5*(1.5-fx)**2, 0.75-(fx-1)**2, 0.5*(fx-0.5)**2] # Bspline

	for i in ti.static(range(3)):
		for j in ti.static(range(3)):
			offset = ti.Vector([i, j])
			weight = w[i][0] * w[j][1]
			dpos = (offset.cast(float) - fx) * dx
			grid_v[base + offset] += weight * vp
			grid_m[base + offset] += weight


@ti.kernel
def particle_to_grid():

	for k in particle_position:

		pos = particle_position[k]
		vel = particle_velocity[k]

		stagger_u = ti.Vector([0.0, 0.5])
		stagger_v = ti.Vector([0.5, 0.0])

		scatter(velocities_u, weights_u, pos, vel.x, stagger_u)
		scatter(velocities_v, weights_v, pos, vel.y, stagger_v)


@ti.kernel
def grid_norm():

	for k in ti.grouped(velocities_u):
		weight = weights_u[k]
		if weight > 0:
			velocities_u[k] = velocities_u[k] / weight

	for k in ti.grouped(velocities_v):
		weight = weights_v[k]
		if weight > 0:
			velocities_v[k] = velocities_v[k] / weight


@ti.kernel
def apply_gravity():
	for i, j in velocities_v:
		# if not is_solid(i, j-1) and not is_solid(i, j):
			velocities_v[i, j] += -9.8 * dt


@ti.kernel
def solve_divergence():

	for i, j in divergences:
		if not is_solid(i, j):

			v_l = velocities_u[i, j]
			v_r = velocities_u[i+1, j]
			v_d = velocities_v[i, j]
			v_u = velocities_v[i, j+1]

			div = v_r - v_l + v_u - v_d

			if is_solid(i-1, j): 
				div += v_l
			if is_solid(i+1, j): 
				div -= v_r
			if is_solid(i, j-1): 
				div += v_d
			if is_solid(i, j+1): 
				div -= v_u

			divergences[i, j] = div / (dx)


@ti.kernel
def pressure_jacobi(p:ti.template(), new_p:ti.template()):

	w = jacobi_damped_para

	for i, j in p:
		if is_fluid(i, j):

			v_l = velocities_u[i, j]
			v_r = velocities_u[i+1, j]
			v_d = velocities_v[i, j]
			v_u = velocities_v[i, j+1]

			p_l = p[i-1, j]
			p_r = p[i+1, j]
			p_d = p[i, j-1]
			p_u = p[i, j+1]


			k = 4
			if is_solid(i-1, j):
				p_l = 0.0
				k -= 1
			if is_solid(i+1, j):
				p_r = 0.0
				k -= 1
			if is_solid(i, j-1):
				p_d = 0.0
				k -= 1
			if is_solid(i, j+1):
				p_u = 0.0
				k -= 1


			new_p[i, j] = (1 - w) * p[i, j] + w * ( p_l + p_r + p_d + p_u - divergences[i, j] * rho / dt * (dx*dx) ) / k


@ti.kernel
def projection():

	for i, j in ti.ndrange(m_g, m_g):
		if is_fluid(i-1, j) or is_fluid(i, j):
			if is_solid(i-1, j) or is_solid(i, j):
				velocities_u[i, j] = 0.0
			else:
				velocities_u[i, j] -= (pressures[i, j] - pressures[i-1, j]) / dx / rho * dt

		if is_fluid(i, j-1) or is_fluid(i, j):
			if is_solid(i, j-1) or is_solid(i, j):
				velocities_v[i, j] = 0.0
			else:
				velocities_v[i, j] -= (pressures[i, j] - pressures[i, j-1]) / dx / rho * dt


@ti.func
def gather(grid_v, last_grid_v, xp, stagger):
	base = (xp * inv_dx - (stagger + 0.5)).cast(ti.i32)
	fx = xp * inv_dx - (base.cast(ti.f32) + stagger)

	w = [0.5*(1.5-fx)**2, 0.75-(fx-1)**2, 0.5*(fx-0.5)**2] # Bspline

	v_pic = 0.0
	v_flip = 0.0

	for i in ti.static(range(3)):
		for j in ti.static(range(3)):
			offset = ti.Vector([i, j])
			weight = w[i][0] * w[j][1]
			v_pic  += weight * grid_v[base + offset]
			v_flip += weight * (grid_v[base + offset] - last_grid_v[base + offset])

	return v_pic, v_flip


@ti.kernel
def grid_to_particle():

	stagger_u = ti.Vector([0.0, 0.5])
	stagger_v = ti.Vector([0.5, 0.0])
	
	for k in ti.grouped(particle_position):
	
		p = particle_position[k]

		pic_x, flip_dx = gather(velocities_u, last_velocities_u, p, stagger_u)
		pic_y, flip_dy = gather(velocities_v, last_velocities_v, p, stagger_v)

		pic_vel = ti.Vector([pic_x, pic_y])
		flip_vel = particle_velocity[k] + ti.Vector([flip_dx, flip_dy])

		particle_velocity[k] = (1-FLIP_blending) * pic_vel + FLIP_blending * flip_vel


@ti.kernel
def advect_particles():

	for k in ti.grouped(particle_position):

		pos = particle_position[k]
		vel = particle_velocity[k]
		
		pos += vel * dt

		if pos.x < dx * boundary_width:
			pos.x = dx * boundary_width
			vel.x = 0
		if pos.x >= length - dx * boundary_width:
			pos.x = length - dx * boundary_width - eps
			vel.x = 0

		if pos.y < dx * boundary_width:
			pos.y = dx * boundary_width
			vel.y = 0
		if pos.y >= length - dx * boundary_width:
			pos.y = length - dx * boundary_width - eps
			vel.y = 0


		particle_position[k] = pos
		particle_velocity[k] = vel



def step():

	init_step()
	mark_cell()

	particle_to_grid()
	grid_norm()

	last_velocities_u.copy_from(velocities_u)
	last_velocities_v.copy_from(velocities_v)

	apply_gravity()
	handle_boundary()


	solve_divergence()
	
	for i in range(jacobi_iters):
		global pressures, new_pressures
		pressure_jacobi(pressures, new_pressures)
		pressures, new_pressures = new_pressures, pressures

	projection()


	grid_to_particle()
	advect_particles()







init_grid()
init_particle()


gui = ti.GUI("FLIP Blending", (res, res))


# result_dir = "./result"
# video_manager = ti.VideoManager(output_dir=result_dir, framerate=30, automatic_build=False)


for frame in range(30000):

	gui.clear(0xFFFFFF)

	for i in range(substep):
		step()


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
				gui.circle([(i+0.5)/m_g, (j+0.5)/m_g], radius = 2, color = color)
				# gui.line([i*dx, j*dx], [i*dx, (j+1)*dx], color = 0xFF0000)
				# gui.line([i*dx, (j+1)*dx], [(i+1)*dx, (j+1)*dx], color = 0xFF0000)
				# gui.line([(i+1)*dx, j*dx], [(i+1)*dx, (j+1)*dx], color = 0xFF0000)
				# gui.line([(i+1)*dx, j*dx], [i*dx, j*dx], color = 0xFF0000)

	gui.circles(particle_position.to_numpy() / length, radius=0.8, color=0x3399FF)

	# video_manager.write_frame(gui.get_image())	
	gui.show()



# video_manager.make_video(gif=True, mp4=True)