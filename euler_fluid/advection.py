import taichi as ti
import numpy
import time



ti.init(arch=ti.cpu)


n = 400
dt = 0.05

# 1, 2, 3
RK = 3

#
enable_BFECC = True

#
enable_clipping = True

pixels = ti.Vector(3, dt=ti.f32, shape=(n, n))
new_pixels = ti.Vector(3, dt=ti.f32, shape=(n, n))
new_new_pixels = ti.Vector(3, dt=ti.f32, shape=(n, n))


# screen center
center = ti.Vector([0.5, 0.5])

# cell center
stagger = ti.Vector([0.5, 0.5])


@ti.func
def I(i, j):
	return ti.Vector([i, j])


@ti.func
def vel(p):
	# rotation
	return ti.Vector([p.y-center.y, center.x-p.x])


@ti.kernel
def init_image():
	# super sampling
	for i, j in ti.ndrange(4*n, 4*n):
		d = ti.Vector([i/4/n, j/4/n]) - center
		if d.norm_sqr() < 0.2:
			x = i/4/n
			if x < 1/3:
				pixels[i//4, j//4] += ti.Vector([0.0, 0.0, 0.0])/16
			elif x < 2/3:
				pixels[i//4, j//4] += ti.Vector([1.0, 1.0, 1.0])/16
			else:
				pixels[i//4, j//4] += ti.Vector([1.0, 0.0, 0.0])/16
		else:
			pixels[i//4, j//4] += ti.Vector([1.0, 1.0, 1.0])/16


@ti.func
def clamp(p):
	# clamp p to [0.5/n, 1-1/n+0.5/n), i.e. clamp cell index to [0, n-1)
	for d in ti.static(range(p.n)):
		p[d] = min(1 - 1/n + stagger[d]/n - 1e-4, max(p[d], stagger[d]/n))
	return p

@ti.func
def sample_bilinear(pixels, p):

	p = clamp(p)

	grid_f = p * n - stagger
	grid_i = ti.cast(ti.floor(grid_f), ti.i32)

	d = grid_f - grid_i
	
	return pixels[ grid_i ] * (1-d.x)*(1-d.y) + pixels[ grid_i+I(1, 0) ] * d.x*(1-d.y) + pixels[ grid_i+I(0, 1) ] * (1-d.x)*d.y + pixels[ grid_i+I(1, 1) ] * d.x*d.y

@ti.func
def sample_min(pixels, p):

	p = clamp(p)

	grid_f = p * n - stagger
	grid_i = ti.cast(ti.floor(grid_f), ti.i32)
	

	color = ti.Vector([0.0, 0.0, 0.0])

	for d in ti.static(range(color.n)):
		color[d] = min( pixels[ grid_i ][d], pixels[ grid_i+I(1, 0) ][d], pixels[ grid_i+I(0, 1) ][d], pixels[ grid_i+I(1, 1) ][d] )

	return color

@ti.func
def sample_max(pixels, p):

	p = clamp(p)

	grid_f = p * n - stagger
	grid_i = ti.cast(ti.floor(grid_f), ti.i32)
	
	color = ti.Vector([0.0, 0.0, 0.0])

	for d in ti.static(range(color.n)):
		color[d] = max( pixels[ grid_i ][d], pixels[ grid_i+I(1, 0) ][d], pixels[ grid_i+I(0, 1) ][d], pixels[ grid_i+I(1, 1) ][d] )

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
def semi_lagrangian(pixels, new_pixels, dt):
	for i in ti.grouped(pixels):
		p = (i + stagger) / n
		new_pixels[i] = sample_bilinear(pixels, backtrace(p, dt))



@ti.func
def BFECC():
	
	semi_lagrangian(pixels, new_pixels, dt)
	semi_lagrangian(new_pixels, new_new_pixels, -dt)

	for i in ti.grouped(pixels):
		
		new_pixels[i] = new_pixels[i] - 0.5 * (new_new_pixels[i] - pixels[i])

		if ti.static(enable_clipping):
			
			source_pos = backtrace( (i + stagger) / n, dt )
			mi = sample_min(pixels, source_pos)
			mx = sample_max(pixels, source_pos)

			if new_pixels[i].x < mi.x or new_pixels[i].y < mi.y or new_pixels[i].z < mi.z or new_pixels[i].x > mx.x or new_pixels[i].y > mx.y or new_pixels[i].z > mx.z:
				new_pixels[i] = sample_bilinear(pixels, source_pos)
				
				# runtime error
				# break



@ti.kernel
def advect():
	if ti.static(enable_BFECC):
		BFECC()
	else:
		semi_lagrangian(pixels, new_pixels, dt)
		
	for i in ti.grouped(pixels):
		pixels[i] = new_pixels[i]



init_image()


gui = ti.GUI("Fluid 2D", (n, n))


result_dir = "./result"
video_manager = ti.VideoManager(output_dir=result_dir, framerate=30, automatic_build=False)

for frame in range(300):
	
	advect()
	# time.sleep(1)

	gui.set_image(pixels)

	gui.text(content=f'RK {RK}', pos=(0, 0.98), color=0x0)
	if enable_BFECC: 
		gui.text(content=f'BFECC', pos=(0, 0.94), color=0x0)
		if enable_clipping: 
			gui.text(content=f'Clipped', pos=(0, 0.90), color=0x0)
	
	# video_manager.write_frame(gui.get_image())
	gui.show()


# video_manager.make_video(gif=True, mp4=True)