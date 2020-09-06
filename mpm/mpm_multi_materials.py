import taichi as ti
import numpy as np
import time


ti.init(arch=ti.gpu, default_fp=ti.f32)


res = 512
dt = 1e-4
substep = int(2e-3 // dt)


m_g = 128
n_grid = m_g*m_g
n_particle = 9000

length = 1.0
dx = length/m_g
inv_dx = 1/dx

# solid boundary
boundary_width = 2

eps = 1e-5


p_rho = 1
p_vol = (dx*0.5)**2
p_mass = p_vol * p_rho

g = -9.8 * 5

# Young's modulus
E = 1e3 

# Poisson's ratio
nu = 0.2

# Lame parameters
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1+nu) * (1 - 2 * nu))


# cell-centered grid

# node momentum
grid_velocity = ti.Vector(2, dt=ti.f32, shape=(m_g, m_g))

# node mass
grid_weight = ti.var(dt=ti.f32, shape=(m_g, m_g))

particle_velocity = ti.Vector(2, dt=ti.f32, shape=n_particle)
particle_position = ti.Vector(2, dt=ti.f32, shape=n_particle)

# affine velocity field
particle_C = ti.Matrix(2, 2, dt=ti.f32, shape=n_particle)

# deformation gradient
particle_F = ti.Matrix(2, 2, dt=ti.f32, shape=n_particle)

# plastic deformation
particle_J = ti.var(dt=ti.f32, shape=n_particle)

# 0: fluid
# 1: jelly
# 2: snow
particle_material = ti.var(dt=ti.i32, shape=n_particle)


#TODO
@ti.kernel
def init_particle():
	
	group_size = n_particle//3
	
	for i in particle_position:
		
		group_id = i // group_size
		particle_material[i] = group_id
		
		particle_position[i] = (ti.Vector([ti.random(), ti.random()])*0.25 + 0.1 + group_id*0.26) * length
		particle_velocity[i] = ti.Vector([0.0, -1.0])

		particle_F[i] = ti.Matrix([[1, 0], [0, 1]])
		particle_J[i] = 1.0


@ti.kernel
def init_step():

	for k in ti.grouped(grid_velocity):
		grid_velocity[k] = ti.Vector([0.0, 0.0])
		grid_weight[k] = 0.0


@ti.kernel
def particle_to_grid():

	for k in particle_position:

		p = particle_position[k]
		grid = p * inv_dx
		base = int(grid-0.5)
		fx = grid - base

		w = [0.5*(1.5-fx)**2, 0.75-(fx-1)**2, 0.5*(fx-0.5)**2] # B-spline
		# stress = -dt * 4 * E * p_vol * (particle_J[k] - 1) * (inv_dx**2)
		# affine = ti.Matrix([[stress, 0], [0, stress]]) + p_mass * particle_C[k]

		particle_F[k] = (ti.Matrix.identity(float, 2) + dt * particle_C[k]) @ particle_F[k]

		# hardening coefficient
		h = ti.exp(10 * (1.0 - particle_J[k]))
		if particle_material[k] == 1: # jelly
			h = 0.3

		mu, la = mu_0 * h, lambda_0 * h
		if particle_material[k] == 0: # fluid
			mu = 0.0
		
		U, sig, V = ti.svd(particle_F[k])
		J = 1.0
		for d in ti.static(range(2)):
			new_sig = sig[d, d]
			if particle_material[k] == 2:  # snow
				new_sig = min(max(sig[d, d], 1 - 2.5e-2), 1 + 4.5e-3)  # plasticity
			particle_J[k] *= sig[d, d] / new_sig
			sig[d, d] = new_sig
			J *= new_sig

		if particle_material[k] == 0:  # Fluid: Reset deformation gradient to avoid numerical instability
			particle_F[k] = ti.Matrix.identity(float, 2) * ti.sqrt(J)
		elif particle_material[k] == 2:
			particle_F[k] = U @ sig @ V.transpose() # Snow: Reconstruct elastic deformation gradient after plasticity

		stress = 2 * mu * (particle_F[k] - U @ V.transpose()) @ particle_F[k].transpose() + ti.Matrix.identity(float, 2) * la * J * (J - 1)
		stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
		affine = stress + p_mass * particle_C[k]

		for i in ti.static(range(3)):
			for j in ti.static(range(3)):
				offset = ti.Vector([i, j])
				weight = w[i][0] * w[j][1]
				dpos = (offset - fx) * dx
				grid_velocity[base + offset] += weight * (p_mass * particle_velocity[k] + affine @ dpos)
				grid_weight  [base + offset] += weight * p_mass


@ti.kernel
def grid_normalization():

	for k in ti.grouped(grid_velocity):
		weight = grid_weight[k]
		if weight > 0:
			grid_velocity[k] = grid_velocity[k] / weight


@ti.kernel
def apply_gravity():

	for k in ti.grouped(grid_velocity):
		grid_velocity[k].y += g * dt

@ti.kernel
def handle_boundary():

	for i, j in grid_velocity:
		if i < boundary_width and grid_velocity[i, j].x < 0:
			grid_velocity[i, j].x = 0
		if i > m_g - 1 - boundary_width and grid_velocity[i, j].x > 0:
			grid_velocity[i, j].x = 0
		if j < boundary_width and grid_velocity[i, j].y < 0:
			grid_velocity[i, j].y = 0
		if j > m_g - 1 - boundary_width and grid_velocity[i, j].y > 0:
			grid_velocity[i, j].y = 0


@ti.kernel
def grid_to_particle():

	for k in particle_position:

		p = particle_position[k]
		grid = p * inv_dx
		base = int(grid-0.5)
		fx = grid - base

		w = [0.5*(1.5-fx)**2, 0.75-(fx-1)**2, 0.5*(fx-0.5)**2] # B-spline

		new_v = ti.Vector([0.0, 0.0])
		new_C = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])

		for i in ti.static(range(3)):
			for j in ti.static(range(3)):
				offset = ti.Vector([i, j])
				weight = w[i][0] * w[j][1]
				dpos = (offset - fx) * dx
				new_v +=     weight * grid_velocity[base + offset]
				new_C += 4 * weight * grid_velocity[base + offset].outer_product(dpos) * (inv_dx**2)

		particle_velocity[k] = new_v
		particle_position[k] += particle_velocity[k] * dt
		# particle_J[k] *= 1 + new_C.trace() * dt
		particle_C[k] = new_C


def step():

	init_step()

	particle_to_grid()
	grid_normalization()

	apply_gravity()
	handle_boundary()

	grid_to_particle()



##############################

init_particle()


gui = ti.GUI("MPM Multi Materials", (res, res))


result_dir = "./result"
video_manager = ti.VideoManager(output_dir=result_dir, framerate=30, automatic_build=False)

for frame in range(300):

	gui.clear(0x0)

	for i in range(substep):
		step()


	colors = np.array([0x3399FF, 0xED553B, 0xFFFFFF], dtype=np.uint32)
	gui.circles(particle_position.to_numpy() / length, radius=1.2, color=colors[particle_material.to_numpy()])

	video_manager.write_frame(gui.get_image())
	gui.show()



video_manager.make_video(gif=True, mp4=True)