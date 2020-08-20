import taichi as ti
import numpy
import time



ti.init(arch=ti.cpu)


n = 512
dt = 0.05

# 1, 2, 3
RK = 3

# enable BFECC
BFECC = False

pixels = ti.Vector(3, dt=ti.f32, shape=(n, n))
new_pixels = ti.Vector(3, dt=ti.f32, shape=(n, n))
new_new_pixels = ti.Vector(3, dt=ti.f32, shape=(n, n))


res = ti.Vector([n, n])
center = ti.Vector([0.5, 0.5])


@ti.func
def I(i, j):
	return ti.Vector([i, j])


@ti.func
def vel(p):
	# rotation
	return ti.Vector([p.y-center.y, center.x-p.x])


@ti.kernel
def init_image():
	for i, j in ti.ndrange(4*n, 4*n):
		d = ti.Vector([i/4/n, j/4/n]) - ti.Vector([0.7, 0.7])
		if d.norm_sqr() < 0.005:
			# pixels[i//4, j//4] = ti.Vector([ti.random(), ti.random(), ti.random()])
			pixels[i//4, j//4] += ti.Vector([0.0, 0.0, 0.0])/16
			# x = i/4/n
			# # pixels[i, j] = ti.Vector([(i//200+1)*0.2, (j//200+1)*0.2, 0.1])
			# if x < 1/3:
			# 	pixels[i//4, j//4] += ti.Vector([1.0, 1.0, 1.0])/16
			# elif x < 2/3:
			# 	pixels[i//4, j//4] += ti.Vector([1.0, 1.0, 1.0])/16
			# else:
			# 	pixels[i//4, j//4] += ti.Vector([0.0, 0.0, 0.0])/16
		else:
			pixels[i//4, j//4] += ti.Vector([1.0, 1.0, 1.0])/16



@ti.func
def bilinear_interpolate(pixels, p):

	# clamp to [0, 1-1/n)
	for i in ti.static(range(p.n)):
		p[i] = min(1-1e-4-1/n, max(p[i], 0))
	
	grid_f = p * res
	grid_i = ti.cast(ti.floor(grid_f), ti.i32)
	d = grid_f - grid_i

	return pixels[ grid_i ] * (1-d.x)*(1-d.y) + pixels[ grid_i+I(1, 0) ] * d.x*(1-d.y) + pixels[ grid_i+I(0, 1) ] * (1-d.x)*d.y + pixels[ grid_i+I(1, 1) ] * d.x*d.y



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
def semi_lagrangian(pixels, new_pixels, dt):
	for i, j in pixels:
		p = ti.Vector([i/n, j/n])
		new_pixels[i, j] = bilinear_interpolate(pixels, backtrace(p, dt))



@ti.func
def BFECC(p):
	for i, j in pixels:
		semi_lagrangian(pixels, new_pixels, p, dt)
		semi_lagrangian(new_pixels, new_new_pixels, p1, -dt)

	for i in ti.grouped(pixels):
		new_pixels[i] = new_pixels[i] + 0.5 * (new_new_pixels[i] - pixels[i])



@ti.kernel
def advect():
	if ti.static(BFECC == True):
		BFECC()
	else:
		semi_lagrangian(pixels, new_pixels, dt)
		
	for i in ti.grouped(pixels):
		pixels[i] = new_pixels[i]



init_image()


gui = ti.GUI("Fluid 2D", (n, n))

pause = False


result_dir = "./fluid_2d"
video_manager = ti.VideoManager(output_dir=result_dir, framerate=30, automatic_build=False)
for frame in range(150):
	
	if not pause:
		advect()
		# pause = True
	# time.sleep(1)

	gui.set_image(pixels)
	gui.show()

	video_manager.write_frame(pixels.to_numpy())

video_manager.make_video(gif=True, mp4=True)