import taichi as ti
import numpy as np
import time



ti.init(arch=ti.gpu)


n = 512
dt = 0.03
# dt = 1/60.0
dx = 1.0


rho = 1
jacobi_iters = 20
make_video = False
# enabl_test = False


dye_decay = 1



colors = ti.Vector(3, dt=ti.f32, shape=(n, n))
new_colors = ti.Vector(3, dt=ti.f32, shape=(n, n))
new_new_colors = ti.Vector(3, dt=ti.f32, shape=(n, n))

velocities = ti.Vector(2, dt=ti.f32, shape=(n, n))
new_velocities = ti.Vector(2, dt=ti.f32, shape=(n, n))
new_new_velocities = ti.Vector(2, dt=ti.f32, shape=(n, n))

pressures = ti.var(dt=ti.f32, shape=(n, n))
new_pressures = ti.var(dt=ti.f32, shape=(n, n))

divergences = ti.var(dt=ti.f32, shape=(n, n))
# curls = ti.var(dt=ti.f32, shape=(n, n))

collisons = ti.var(dt=ti.f32, shape=(n, n))

center = ti.Vector([0.5, 0.5]) * n
lower_center = ti.Vector([0.5, 0.0]) * n

# cell center
stagger = ti.Vector([0.5, 0.5])

collision_center = ti.Vector([0.1, 0.5]) * n
collision_radius = 10

@ti.kernel
def init(collisons:ti.template(), velocities:ti.template(), colors:ti.template()):
	for i in ti.grouped(collisons):
		#collsion
		d = i - collision_center
		if d.norm() < collision_radius:
			collisons[i] = 1
		else:
			collisons[i] = 0

	for i in ti.grouped(velocities):
		velocities[i] = ti.Vector([100, 0])

@ti.kernel
def init_preframe(velocities:ti.template(), colors:ti.template()):
	for i in ti.grouped(colors):
		color = ti.Vector([1.0, 1.0, 1.0])
		if i.y > n/2:
			color = ti.Vector([0.0, 0.5, 0.9])
		else:
			color = ti.Vector([0.7, 0.3, 0.2])
		if i.x < 20:
			if ti.cos(i.y*200/n) > 0.5:
				colors[i] = color
			else:
				colors[i] = ti.Vector([0, 0, 0])

@ti.kernel
def process_collision(velocities:ti.template(), colors:ti.template(), collisons:ti.template()):
	for i in ti.grouped(collisons):
		#collsion
		if collisons[i] > 0:
			velocities[i] = ti.Vector([0, 0])
			colors[i] = ti.Vector([0, 0, 0])

@ti.func
def I(i, j):
	return ti.Vector([i, j])


@ti.func
def sample(field, p):
	grid_i = ti.Vector([int(p.x), int(p.y)])
	grid_i = max(0, min(n - 1, grid_i))

	return field[ grid_i ]

@ti.func
def lerp(vl, vr, frac):
	# frac: [0.0, 1.0]
	return vl + frac * (vr - vl)

@ti.func
def sample_bilinear(field, p):

	grid_f = p - stagger
	grid_i = ti.Vector([int(grid_f.x), int(grid_f.y)])

	df = grid_f - grid_i

	a = sample(field, grid_i)
	b = sample(field, grid_i+I(1, 0))
	c = sample(field, grid_i+I(0, 1))
	d = sample(field, grid_i+I(1, 1))

	return lerp(lerp(a, b, df.x), lerp(c, d, df.x), df.y)
	

@ti.func
def backtrace(vel_field, p, dt):
	# return p - sample_bilinear(vel_field, p) * dt
	k1 = sample_bilinear(velocities, p)
	k2 = sample_bilinear(velocities, p - k1 * dt * 0.5)
	k3 = sample_bilinear(velocities, p - k2 * dt * 0.5)
	k4 = sample_bilinear(velocities, p - k3 * dt)
	return p - (0.5*k1 + k2 + k3 + 0.5*k4) * dt / 3.0
@ti.func
def semi_lagrangian(vel_field, field, new_field, dt):
	for i in ti.grouped(field):
		p = (i + stagger) * dx
		p = backtrace(vel_field, p, dt)
		new_field[i] = sample_bilinear(field, p)


@ti.kernel
def advect(vel_field:ti.template(), field:ti.template(), new_field:ti.template(), dt:ti.f32):
	semi_lagrangian(vel_field, field, new_field, dt)


@ti.kernel
def solve_divergence(velocities:ti.template(), divergences:ti.template()):
	for i, j in velocities:


		if i == 0: 
			velocities[i, j] = ti.Vector([abs(velocities[i, j].x), 0])
		if i == n-1:
			velocities[i, j] = ti.Vector([abs(velocities[i, j].x), 0])
		if j == 0:
			velocities[i, j] = ti.Vector([abs(velocities[i, j].x), 0])
		if j == n-1:
			velocities[i, j] = ti.Vector([abs(velocities[i, j].x), 0])


		c = ti.Vector([i, j]) * dx
		l = c + ti.Vector([-1, 0]) * dx
		r = c + ti.Vector([1, 0]) * dx
		d = c + ti.Vector([0, -1]) * dx
		u = c + ti.Vector([0, 1]) * dx
		v_c = sample(velocities, c)
		v_l = sample(velocities, l).x
		v_r = sample(velocities, r).x
		v_d = sample(velocities, d).y
		v_u = sample(velocities, u).y


		divergences[i, j] = (v_r - v_l + v_u - v_d) / (2*dx)



@ti.kernel
def pressure_jacobi(pressures:ti.template(), new_pressures:ti.template()):

	for i, j in pressures:
		c = ti.Vector([i, j]) * dx
		l = c + ti.Vector([-1, 0]) * dx
		r = c + ti.Vector([1, 0]) * dx
		d = c + ti.Vector([0, -1]) * dx
		u = c + ti.Vector([0, 1]) * dx

		p_l = sample(pressures, l)
		p_r = sample(pressures, r)
		p_d = sample(pressures, d)
		p_u = sample(pressures, u)

		new_pressures[i, j] = ( p_l + p_r + p_d + p_u - divergences[i, j] * rho  * (dx*dx) ) * 0.25



@ti.kernel
def projection(velocities:ti.template(), pressures:ti.template()):
	for i, j in velocities:
		c = ti.Vector([i, j]) * dx
		l = c + ti.Vector([-1, 0]) * dx
		r = c + ti.Vector([1, 0]) * dx
		d = c + ti.Vector([0, -1]) * dx
		u = c + ti.Vector([0, 1]) * dx
		p_c = sample(pressures, c)
		p_l = sample(pressures, l)
		p_r = sample(pressures, r)
		p_d = sample(pressures, d)
		p_u = sample(pressures, u)

		velocities[i, j] -= ti.Vector([p_r - p_l, p_u - p_d]) / (2*dx) / rho


@ti.kernel
def apply_force(velocities:ti.template(), colors:ti.template(), collisons:ti.template(), imp_data: ti.ext_arr()):

	f_strength = 10000.0
	force_radius = n / 3.0
	for i, j in velocities:
		omx, omy = imp_data[2], imp_data[3]
		mdir = ti.Vector([imp_data[0], imp_data[1]])
		dx, dy = (i + 0.5 - omx), (j + 0.5 - omy)
		d2 = dx * dx + dy * dy
		# dv = F * dt
		factor = ti.exp(-d2 / force_radius)
		momentum = mdir * f_strength * dt * factor
		v = velocities[i, j]
		velocities[i, j] = v + momentum
		# add dye
		color = ti.Vector([1.0, 1.0, 1.0])
		dc = colors[i, j]
		if mdir.norm() > 0.5:
			# dc += ti.exp(-d2 * (4 / (n / 15)**2)) * ti.Vector([imp_data[4], imp_data[5], imp_data[6]])
			dc += ti.exp(-d2 * (4 / (n / 15)**2)) * color
		dc *= dye_decay
		colors[i, j] = dc

		if mdir.norm() > 0.5 and d2 < 50:
			collisons[i, j] = 1


class MouseDataGen(object):
	def __init__(self):
		self.prev_mouse = None
		self.prev_color = None

	def __call__(self, gui, frame):
		# [0:2]: normalized delta direction
		# [2:4]: current mouse xy
		# [4:7]: color
		mouse_data = np.zeros(8, dtype=np.float32)
		if gui.is_pressed(ti.GUI.LMB):
			mxy = np.array(gui.get_cursor_pos(), dtype=np.float32) * n
			if self.prev_mouse is None:
				self.prev_mouse = mxy
				# Set lower bound to 0.3 to prevent too dark colors
				self.prev_color = (np.random.rand(3) * 0.7) + 0.3
			else:
				mdir = mxy - self.prev_mouse
				mdir = mdir / (np.linalg.norm(mdir) + 1e-5)
				mouse_data[0], mouse_data[1] = mdir[0], mdir[1]
				mouse_data[2], mouse_data[3] = mxy[0], mxy[1]
				mouse_data[4:7] = self.prev_color
				self.prev_mouse = mxy
		else:
			self.prev_mouse = None
			self.prev_color = None
		return mouse_data


gui = ti.GUI("Fluid 2D", (n, n))
gui.fps_limit = 60

md_gen = MouseDataGen()

pre_mouse_pos = None
cur_mouse_pos = None
cur_mouse_dir = None

result_dir = "./result"
video_manager = ti.VideoManager(output_dir=result_dir, framerate=60, automatic_build=False)

init(collisons, velocities, colors)

for frame in range(1000):



	gui.get_event(ti.GUI.PRESS)

	mouse_data = md_gen(gui, frame)

	advect(velocities, velocities, new_velocities, dt)
	advect(velocities, colors, new_colors, dt)
	velocities, new_velocities = new_velocities, velocities
	colors, new_colors = new_colors, colors

	init_preframe(velocities, colors)



	apply_force(velocities, colors, collisons, mouse_data)

	process_collision(velocities, colors, collisons)

	solve_divergence(velocities, divergences)



	for i in range(jacobi_iters):
		pressure_jacobi(pressures, new_pressures)
		pressures, new_pressures = new_pressures, pressures

	# projection()
	projection(velocities, pressures)



	gui.set_image(colors)
	# gui.set_image(divergences.to_numpy() * 0.1 + 0.5)
	# gui.set_image(pressures.to_numpy() * 0.001 + 0.5)

	# gui.text(content=f'Frame{frame}', pos=(0, 0.5), color=0xFFFFFF)

	# video_manager.write_frame(gui.get_image())
	video_manager.write_frame(colors.to_numpy())
	gui.show()



# video_manager.make_video(gif=True, mp4=True)

