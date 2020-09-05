# http://www.eng.fsu.edu/~dommelen/papers/subram/style_a/node20.html

import taichi as ti
import numpy as np
import math

ti.init(arch=ti.gpu)


width = 1600
height = 600

eps = 0.01
dt = 0.1

particle_num = 200000
vortex_num = 6

particles = ti.Vector(2, dt=ti.f32, shape=particle_num)
colors = ti.Vector(3, dt=ti.f32, shape=particle_num)

vortex = ti.Vector(2, dt=ti.f32, shape=vortex_num)
new_vortex = ti.Vector(2, dt=ti.f32, shape=vortex_num)
sign = ti.var(dt=ti.i32, shape=vortex_num)


pixels = ti.Vector(3, dt=ti.f32, shape=(width, height))



@ti.func
def compute_u(p, i):
  r2 = (p-vortex[i]).norm_sqr()
  uv = ti.Vector([vortex[i].y-p.y, p.x-vortex[i].x])
  return sign[i] * uv / (r2 * math.pi) * 0.5 * (1.0 - ti.exp(-r2/eps**2))


# @ti.kernel
# def advect_particle():
#   for i in range(particle_num):
#     p = particles[i]
#     for j in range(vortex_num):
#       particles[i] += compute_u(p, j) * dt


@ti.kernel
def advect_particle():
  for i in range(particle_num):
    p = particles[i]
    v1 = ti.Vector([0.0, 0.0])
    v2 = ti.Vector([0.0, 0.0])
    v3 = ti.Vector([0.0, 0.0])
    for j in range(vortex_num):
      v1 += compute_u(p, j)
      v2 += compute_u(p + v1 * dt * 0.5, j)
      v3 += compute_u(p + v2 * dt * 0.75, j)
    particles[i] += (2/9 * v1 + 1/3 * v2 + 4/9 * v3) * dt



@ti.kernel
def integrate_vortex():
  for i in range(vortex_num):
  	v = ti.Vector([0.0, 0.0]) # 0.0 not 0 !!
  	for j in range(vortex_num):
  		if i != j :
  			v += compute_u(vortex[i], j)
  	new_vortex[i] = vortex[i] + v * dt

  for i in range(vortex_num):
    vortex[i] = new_vortex[i]



@ti.kernel
def init():
  vortex[0] = ti.Vector([0, 1])
  vortex[1] = ti.Vector([0, -1])
  vortex[2] = ti.Vector([0, 0.3])
  vortex[3] = ti.Vector([0, -0.3])
  vortex[4] = ti.Vector([0, 0.65])
  vortex[5] = ti.Vector([0, -0.65])
 
  sign[0] = 1
  sign[1] = -1
  sign[2] = 1
  sign[3] = -1
  sign[4] = 1
  sign[5] = -1

  for i in range(particle_num):
    x = ti.random()
    y = ti.random()
    particles[i] = ti.Vector([x*3-1.5, y-0.5])
    # colors[i] = ti.Vector([x, y, y])
    colors[i] = ti.Vector([x+0.3, y+0.3, 0.0])


@ti.kernel
def get_image():

  for i in ti.grouped(pixels):
    pixels[i]=ti.Vector([0.2, 0.2, 0.2])
  
  for i in range(particle_num):
  	pos = particles[i] * ti.Vector([0.1*height/width, 0.1]) + ti.Vector([0.0, 0.5])
  	win_pos = pos * ti.Vector([width, height])
  	if win_pos.x >= 0 and win_pos.x < width and win_pos.y >= 0 and win_pos.y < height:
  		pixels[int(win_pos.x), int(win_pos.y)] = colors[i]





init()


gui=ti.GUI("Vortex Leapfrogging", (width, height), 0x0)

result_dir = "./result"
video_manager = ti.VideoManager(output_dir=result_dir, framerate=30, automatic_build=False)


for frame in range(600):

  for i in range(4):
    advect_particle()
    integrate_vortex()

  # too slow
  # for i in range(particle_num):
  #   gui.circles(np.array([[particles[i].x, particles[i].y]]) * np.array([[0.1*height/width, 0.1]]) + np.array([[0.0, 0.5]]), radius=0.5, color=0x0000FF)

  get_image()
  gui.set_image(pixels)
  # gui.circles(particles.to_numpy() * np.array([[0.1*height/width, 0.1]]) + np.array([[0.0, 0.5]]), radius=0.5, color=0x000000)
  gui.show()

  video_manager.write_frame(pixels.to_numpy())


video_manager.make_video(gif=True, mp4=True)
