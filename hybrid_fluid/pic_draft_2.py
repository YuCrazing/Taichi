import taichi as ti
import numpy as np
import time


# apply force after grid normalization (or explosion)
# pressure in air cells should be 0 (or volume shrink quickly)
# velocity in air cells may not be 0

# FIX: adjusted the boundary response in handle_boundary() and advect_particles(), is seems to be more energetic now

# TODO: solve vorticity enhancement explosion problem

ti.init(arch=ti.gpu, default_fp=ti.f32)


res = 512
dt = 3e-3 #2e-3 #2e-2
substep = 10

# prevent volume shrinking
extrapolation_iters = 10

rho = 1000
jacobi_iters = 300
jacobi_damped_para = 0.67



m_g = 128
n_grid = m_g*m_g
n_particle = n_grid*4

length = 1.0
dx = length/m_g
inv_dx = 1/dx

# solid boundary
boundary_width = 2

eps = 1e-5


# boundary condition test
# bounce = 0

# enhance vorticity
curl_strength = 0.0


# show grid types
debug = False


# MAC grid
velocities_u = ti.var(dt=ti.f32, shape=(m_g+1, m_g))
velocities_v = ti.var(dt=ti.f32, shape=(m_g, m_g+1))

weights_u = ti.var(dt=ti.f32, shape=(m_g+1, m_g))
weights_v = ti.var(dt=ti.f32, shape=(m_g, m_g+1))


pressures = ti.var(dt=ti.f32, shape=(m_g, m_g))
new_pressures = ti.var(dt=ti.f32, shape=(m_g, m_g))
divergences = ti.var(dt=ti.f32, shape=(m_g, m_g))
vorticities = ti.var(dt=ti.f32, shape=(m_g, m_g))

### for extrapolation
valid = ti.var(dt=ti.i32, shape=(m_g+1, m_g+1))
valid_tmp = ti.var(dt=ti.i32, shape=(m_g+1, m_g+1))
velocities_u_tmp = ti.var(dt=ti.f32, shape=(m_g+1, m_g))
velocities_v_tmp = ti.var(dt=ti.f32, shape=(m_g, m_g+1))

# visited = ti.var(dt=ti.i32, shape=(m_g+1, m_g+1))
###

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
		particle_position[i] = (ti.Vector([ti.random(), ti.random()])*0.5 + 0.05) * length
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
def handle_boundary(velocities_u:ti.template(), velocities_v:ti.template()):

	for i, j in velocities_u:
		if is_solid(i-1, j) and velocities_u[i, j] < 0:
			velocities_u[i, j] = 0
		if is_solid(i, j) and velocities_u[i, j] > 0:
			velocities_u[i, j] = 0	
	
	for i, j in velocities_v:
		if is_solid(i, j-1) and velocities_v[i, j] < 0:
			velocities_v[i, j] = 0.0
		if is_solid(i, j) and velocities_v[i, j] > 0:
			velocities_v[i, j] = 0.0



@ti.kernel
def init_step(types:ti.template(), particle_position:ti.template(), velocities_u:ti.template(), velocities_v:ti.template(), weights_u:ti.template(), weights_v:ti.template(), pressures:ti.template(), new_pressures:ti.template()):

	for i, j in types:
		if not is_solid(i, j):
			types[i, j] = AIR

	for k in particle_position:
		grid = (particle_position[k] * inv_dx).cast(int)
		if not is_solid(grid.x, grid.y):
			types[grid] = FLUID


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



@ti.func
def scatter(grid_v, grid_m, xp, vp, stagger):
    base = (xp * inv_dx - (stagger + 0.5)).cast(ti.i32)
    fx = xp * inv_dx - (base.cast(ti.f32) + stagger)

    w = [0.5*(1.5-fx)**2, 0.75-(fx-1)**2, 0.5*(fx-0.5)**2] # Bspline

    for i in ti.static(range(3)):
        for j in ti.static(range(3)):
            offset = ti.Vector([i, j])
            weight = w[i][0] * w[j][1]
            grid_v[base + offset] += weight * vp
            grid_m[base + offset] += weight


@ti.kernel
def particle_to_grid(velocities_u:ti.template(), velocities_v:ti.template(), weights_u:ti.template(), weights_v:ti.template(), particle_position:ti.template(), particle_velocity:ti.template()):

	for k in particle_position:

		p = particle_position[k]

		stagger_u = ti.Vector([0.0, 0.5])
		stagger_v = ti.Vector([0.5, 0.0])

		scatter(velocities_u, weights_u, p, particle_velocity[k].x, stagger_u)
		scatter(velocities_v, weights_v, p, particle_velocity[k].y, stagger_v)


@ti.kernel
def grid_normalization(velocities_u:ti.template(), velocities_v:ti.template(), weights_u:ti.template(), weights_v:ti.template()):

	for k in ti.grouped(velocities_u):
		weight = weights_u[k]
		if weight > 0:
			velocities_u[k] = velocities_u[k] / weight

	for k in ti.grouped(velocities_v):
		weight = weights_v[k]
		if weight > 0:
			velocities_v[k] = velocities_v[k] / weight


@ti.kernel
def apply_gravity(velocities_v:ti.template()):
	for i, j in velocities_v:
		velocities_v[i, j] += -9.8 * dt


@ti.kernel
def solve_divergence(velocities_u:ti.template(), velocities_v:ti.template(), divergences:ti.template()):

	for i, j in divergences:
		if not is_solid(i, j):

			v_l = velocities_u[i, j]
			v_r = velocities_u[i+1, j]
			v_d = velocities_v[i, j]
			v_u = velocities_v[i, j+1]

			div = v_r - v_l + v_u - v_d

			divergences[i, j] = div / (dx)

@ti.kernel
def solve_vorticity(velocities_u:ti.template(), velocities_v:ti.template(), vorticities:ti.template()):

	for i, j in vorticities:
		if not is_solid(i, j):

			v_l = 0.25 * (velocities_v[i, j] + velocities_v[i, j+1] + velocities_v[i-1, j] + velocities_v[i-1, j+1])
			v_r = 0.25 * (velocities_v[i, j] + velocities_v[i, j+1] + velocities_v[i+1, j] + velocities_v[i+1, j+1])
			v_d = 0.25 * (velocities_u[i, j] + velocities_u[i+1, j] + velocities_u[i, j-1] + velocities_u[i+1, j-1])
			v_u = 0.25 * (velocities_u[i, j] + velocities_u[i+1, j] + velocities_u[i, j+1] + velocities_u[i+1, j+1])

			vorticities[i, j] = ( (v_r - v_l) - (v_u - v_d) ) / dx

@ti.kernel
def enhance_vorticity(velocities_u:ti.template(), velocities_v:ti.template(), vorticities:ti.template()):

	for i, j in vorticities:
		if not is_solid(i, j):

			vor_l = vorticities[i-1, j]
			vor_r = vorticities[i+1, j]
			vor_d = vorticities[i, j-1]
			vor_u = vorticities[i, j+1]
			vor_c = vorticities[i, j]

			force = ti.Vector([abs(vor_u) - abs(vor_d), abs(vor_l) - abs(vor_r)]).normalized(1e-3)
			force *= curl_strength * vor_c

			velocities_u[i, j] = min(max(velocities_u[i, j] + force.x * dt, -1e3), 1e3)
			velocities_u[i+1, j] = min(max(velocities_u[i+1, j] + force.x * dt, -1e3), 1e3)
			velocities_v[i, j] = min(max(velocities_v[i, j] + force.x * dt, -1e3), 1e3)
			velocities_v[i, j+1] = min(max(velocities_v[i, j+1] + force.x * dt, -1e3), 1e3)


@ti.kernel
def pressure_jacobi(p:ti.template(), new_p:ti.template()):

	w = jacobi_damped_para

	for i, j in p:
		if is_fluid(i, j):

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
def projection(velocities_u:ti.template(), velocities_v:ti.template(), pressures:ti.template()):

	for i, j in ti.ndrange(m_g, m_g):
		if is_fluid(i-1, j) or is_fluid(i, j):
			if is_solid(i-1, j) or is_solid(i, j):
				# velocities_u[i, j] = 0.0
				pass
			else:
				velocities_u[i, j] -= (pressures[i, j] - pressures[i-1, j]) / dx / rho * dt

		if is_fluid(i, j-1) or is_fluid(i, j):
			if is_solid(i, j-1) or is_solid(i, j):
				# velocities_v[i, j] = 0.0
				pass
			else:
				velocities_v[i, j] -= (pressures[i, j] - pressures[i, j-1]) / dx / rho * dt


	# for i, j in velocities_u:
	# 	if is_solid(i-1, j):
	# 		velocities_u[i, j] = -velocities_u[i+1, j] * bounce
	# 	elif is_solid(i, j):
	# 		velocities_u[i, j] = -velocities_u[i-1, j] * bounce

	# for i, j in velocities_v:
	# 	if is_solid(i, j-1):
	# 		velocities_v[i, j] = -velocities_v[i, j+1] * bounce
	# 	elif is_solid(i, j):
	# 		velocities_v[i, j] = -velocities_v[i, j-1] * bounce

	# for i, j in velocities_u:
	# 	if is_solid(i-1, j) or is_solid(i, j):
	# 		velocities[i]
	# 		if velocities_u[i, j] > 0:
	# 			print("----------------", i, j, velocities_u[i, j])
	# for i, j in velocities_v:
	# 	if is_solid(i, j-1) or is_solid(i, j):
	# 		if velocities_v[i, j] > 0:
	# 			print("----------------", i, j, velocities_v[i, j])


@ti.func
def gather(grid_v, xp, stagger):
    base = (xp * inv_dx - (stagger + 0.5)).cast(ti.i32)
    fx = xp * inv_dx - (base.cast(ti.f32) + stagger)

    w = [0.5*(1.5-fx)**2, 0.75-(fx-1)**2, 0.5*(fx-0.5)**2] # Bspline

    vel = 0.0

    for i in ti.static(range(3)):
        for j in ti.static(range(3)):
            offset = ti.Vector([i, j])
            weight = w[i][0] * w[j][1]
            vel += weight * grid_v[base + offset]

    return vel


@ti.kernel
def grid_to_particle(velocities_u:ti.template(), velocities_v:ti.template(), particle_position:ti.template()):

	stagger_u = ti.Vector([0.0, 0.5])
	stagger_v = ti.Vector([0.5, 0.0])
	
	for k in ti.grouped(particle_position):
	
		p = particle_position[k]

		new_u = gather(velocities_u, p, stagger_u)
		new_v = gather(velocities_v, p, stagger_v)

		particle_velocity[k] = ti.Vector([new_u, new_v])



# two methods [clamping velcocity] or [using normal] seem get a similar result
@ti.kernel
def advect_particles(particle_position:ti.template(), particle_velocity:ti.template()):

	for k in ti.grouped(particle_position):

		pos = particle_position[k]
		vel = particle_velocity[k]
		
		pos += vel * dt

		# normal = ti.Vector([0.0, 0.0])

		if pos.x < dx * boundary_width and vel.x < 0:
			pos.x = dx * boundary_width
			vel.x = 0
			# normal.x -= 1.0

		if pos.x >= length - dx * boundary_width and vel.x > 0:
			pos.x = length - dx * boundary_width - eps
			vel.x = 0
			# normal.x += 1.0

		if pos.y < dx * boundary_width and vel.y < 0:
			pos.y = dx * boundary_width
			vel.y = 0
			# normal.y -= 1.0

		if pos.y >= length - dx * boundary_width and vel.y > 0:
			pos.y = length - dx * boundary_width - eps
			vel.y = 0
			# normal.y += 1.0

		# l = normal.norm()
		# if l > eps:
		# 	normal /= l
		# 	vel -= (vel.dot(normal)) * normal


		particle_position[k] = pos
		particle_velocity[k] = vel


@ti.kernel
def mark_valid_u(velocities_u:ti.template(), valid:ti.template()):
	for i, j in velocities_u:
		if is_fluid(i-1, j) or is_fluid(i, j):
			valid[i, j] = 1
		else:
			valid[i, j] = 0

@ti.kernel
def mark_valid_v(velocities_v:ti.template(), valid:ti.template()):
	for i, j in velocities_v:
		if is_fluid(i, j-1) or is_fluid(i, j):
			valid[i, j] = 1
		else:
			valid[i, j] = 0

@ti.kernel
def diffuse(v_new:ti.template(), v:ti.template(), valid_new:ti.template(), valid:ti.template()):
	for i, j in v_new:
		if valid[i, j] == 0:
			s = 0.0
			count = 0
			for m, n in ti.static(ti.ndrange((-1, 2), (-1, 2))):
				if valid[i+m, j+n] == 1:
					s += v[i+m, j+n]
					count += 1
			if count > 0:
				v_new[i, j] = s / count
				valid_new[i, j] = 1

def extrapolate_velocity(velocities_u:ti.template(), velocities_u_tmp:ti.template(), velocities_v:ti.template(), velocities_v_tmp:ti.template(), valid:ti.template(), valid_tmp:ti.template(), iter_num):
	
	mark_valid_u(velocities_u, valid)
	for i in range(iter_num):
		velocities_u_tmp.copy_from(velocities_u)
		valid_tmp.copy_from(valid)
		diffuse(velocities_u, velocities_u_tmp, valid, valid_tmp)

	mark_valid_v(velocities_v, valid)
	for i in range(iter_num):
		velocities_v_tmp.copy_from(velocities_v)
		valid_tmp.copy_from(valid)
		diffuse(velocities_v, velocities_v_tmp, valid, valid_tmp)



def step():

	global types, particle_velocity, particle_position, velocities_u, velocities_v, weights_u, weights_v, pressures, new_pressures, valid, valid_tmp

	init_step(types, particle_position, velocities_u, velocities_v, weights_u, weights_v, pressures, new_pressures)

	particle_to_grid(velocities_u, velocities_v, weights_u, weights_v, particle_position, particle_velocity)
	grid_normalization(velocities_u, velocities_v, weights_u, weights_v)

	apply_gravity(velocities_v)
	handle_boundary(velocities_u, velocities_v)

	extrapolate_velocity(velocities_u, velocities_u_tmp, velocities_v, velocities_v_tmp, valid, valid_tmp, extrapolation_iters)
	# handle_boundary()

	solve_divergence(velocities_u, velocities_v, divergences)

	solve_vorticity(velocities_u, velocities_v, vorticities)
	enhance_vorticity(velocities_u, velocities_v, vorticities)

	for i in range(jacobi_iters):
		# global pressures, new_pressures
		pressure_jacobi(pressures, new_pressures)
		pressures, new_pressures = new_pressures, pressures

	projection(velocities_u, velocities_v, pressures)

	extrapolate_velocity(velocities_u, velocities_u_tmp, velocities_v, velocities_v_tmp, valid, valid_tmp, extrapolation_iters)
	# handle_boundary()

	grid_to_particle(velocities_u, velocities_v, particle_position)
	advect_particles(particle_position, particle_velocity)







init_grid()
init_particle()


gui = ti.GUI("PIC", (res, res))


result_dir = "./result"
video_manager = ti.VideoManager(output_dir=result_dir, framerate=30, automatic_build=False)

for frame in range(300):

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

	gui.text('PIC', pos=(0.05, 0.95), color=0x0)

	# video_manager.write_frame(gui.get_image())
	gui.show()



# video_manager.make_video(gif=True, mp4=True)