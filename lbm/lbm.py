# References:
# Lattice Boltzmann Method for Fluid Simulations
# https://github.com/hietwll/LBM_Taichi/blob/master/lbm_solver.py

import taichi as ti
import numpy as np
import matplotlib
import matplotlib.cm as cm


ti.init(arch=ti.gpu)


@ti.data_oriented
class LBM_Solver:
    def __init__(self, 
                nx, 
                ny, # dx = dy = dt = 1
                niu, 
                bc_type, # [left, right, top, bottom] boundary conditions. 0: Dirichlet, 1: Neumann
                bc_value,
                circle_center,
                circle_radius,
                steps):

        self.nx = nx
        self.ny = ny
        self.niu = niu
        self.tau = 3.0 * niu + 0.5
        self.inv_tau = 1.0 / self.tau
        self.rho = ti.field(dtype=ti.f32, shape=(nx, ny))
        # self.vel = ti.Vector.field(2, dtype=ti.f32, shape=(nx, ny))
        self.vel = ti.Vector(2, dt=ti.f32, shape=(nx, ny))
        # self.f = ti.Vector.field(9, dtype=ti.f32, shape=(nx, ny))
        self.f = ti.Vector(9, dt=ti.f32, shape=(nx, ny))
        # self.f_new = ti.Vector.field(9, dtype=ti.f32, shape=(nx, ny))
        self.f_new = ti.Vector(9, dt=ti.f32, shape=(nx, ny))
        self.w = ti.field(dtype=ti.f32, shape=9)
        # self.e = ti.Vector.field(2, dtype=ti.i32, shape=9)
        self.e = ti.Vector(2, dt=ti.i32, shape=9)
        self.bc_type = ti.field(dtype=ti.i32, shape=4)
        # self.bc_value = ti.Vector.field(2, dtype=ti.f32, shape=4)
        self.bc_value = ti.Vector(2, dt=ti.f32, shape=4)
        self.mask = ti.field(dtype=ti.i32, shape=(nx, ny))
        self.circle_center = ti.Vector(circle_center)*1.0
        self.circle_radius = circle_radius*1.0
        self.steps = steps

        # self.color = ti.Vector(3, dt=ti.f32, shape=(nx, ny))

        # print("niu:", self.niu, "tau:", self.tau)

        self.bc_type.from_numpy(np.array(bc_type, dtype=np.int32))
        self.bc_value.from_numpy(np.array(bc_value, dtype=np.float32))

        self.w.from_numpy(np.array([4.0/9.0,  1.0/9.0,  1.0/9.0,
                                    1.0/9.0,  1.0/9.0,  1.0/36.0,
                                    1.0/36.0, 1.0/36.0, 1.0/36.0], dtype=np.float32))

        self.e.from_numpy(np.array([[0,  0], [1,   0], [0,  1],
                                    [-1, 0], [0,  -1], [1,  1],
                                    [-1, 1], [-1, -1], [1, -1]], dtype=np.int32))

    @ti.func
    def f_eq(self, I, k):
        eu = self.e[k].dot(self.vel[I])
        u2 = self.vel[I].norm_sqr()
        return self.w[k] * self.rho[I] * (1.0 + 3.0 * eu + 4.5 * eu**2 - 1.5 * u2) #???????????????????????????????????


    @ti.func
    def is_boundary(self, I):
        return I.x == 0 or I.x == self.nx-1 or I.y == 0 or I.y == self.ny-1

    @ti.kernel
    def init(self):
        for I in ti.grouped(self.vel):
            self.vel[I] = ti.Vector([0.0, 0.0])
            self.rho[I] = 1.0
            self.mask[I] = 0
            # self.color[I] = ti.Vector([0.0, 0.0, 0.0])
            for k in ti.static(range(9)):
                self.f[I][k] = self.f_eq(I, k)
                self.f_new[I][k] = self.f[I][k]


            d = (self.circle_center - I).norm()
            if d <= self.circle_radius:
                self.mask[I] = 2 # solid collision
            if self.is_boundary(I):
                self.mask[I] = 1 # boundary




    @ti.kernel
    def stream_and_collision(self):
        for I in ti.grouped(self.f):
            if I.x >= 1 and I.x <= self.nx-2 and I.y >= 1 and I.y <= self.ny-2:
                for k in ti.static(range(9)):
                    J = I - self.e[k]
                    self.f_new[I][k] = (1.0 - self.inv_tau) * self.f[J][k] + self.f_eq(J, k) * self.inv_tau


    @ti.kernel
    def compute_macro_var(self):
        for I in ti.grouped(self.vel):
            if I.x >= 1 and I.x <= self.nx-2 and I.y >= 1 and I.y <= self.ny-2:
                self.rho[I] = 0.0
                self.vel[I] = ti.Vector([0.0, 0.0])
                for k in ti.static(range(9)):
                    self.f[I][k] = self.f_new[I][k]
                    self.rho[I] += self.f[I][k]
                    self.vel[I] += self.e[k] * self.f[I][k]
                self.vel[I] /= self.rho[I]


    @ti.kernel
    def apply_bc(self):
        for I in ti.grouped(self.vel):
            
            if self.mask[I]:

                J = I

                if self.mask[I] == 1:
                    
                    # boundary
                    if I.x == 0:
                        J += self.e[1]
                        self.vel[I] = (self.vel[J] if self.bc_type[0] else self.bc_value[0])
                    if I.x == self.nx-1:
                        J += self.e[3]
                        self.vel[I] = (self.vel[J] if self.bc_type[1] else self.bc_value[1])
                    if I.y == 0:
                        J += self.e[2]
                        self.vel[I] = (self.vel[J] if self.bc_type[2] else self.bc_value[2])
                    if I.y == self.ny-1:
                        J += self.e[4]
                        self.vel[I] = (self.vel[J] if self.bc_type[3] else self.bc_value[3])

                elif self.mask[I] == 2:

                    # velocity is zero at solid boundary
                    self.vel[I] = ti.Vector([0.0, 0.0])

                    if I.x < self.circle_center.x:
                        J += self.e[3]
                    else:
                        J += self.e[1]
                    if I.y < self.circle_center.y:
                        J += self.e[4]
                    else:
                        J += self.e[2]


                self.rho[I] = self.rho[J]
                
                for k in ti.static(range(9)):
                    self.f[I][k] = self.f_eq(I, k) - self.f_eq(J, k) + self.f[J][k]


    def solve(self):

        gui = ti.GUI("Lattice Boltzmann Method (D2Q9)", (lbm.nx, lbm.ny))
        gui.fps_limit = None
        result_dir = "./result"
        video_manager = ti.VideoManager(output_dir=result_dir, framerate=60, automatic_build=False)

        lbm.init()

        for i in range(self.steps):

            self.stream_and_collision()
            self.compute_macro_var()
            self.apply_bc()


            # rho = (self.rho.to_numpy()-1.0)*2.0
            
            # mask = self.mask.to_numpy()*1.0

            vel = self.vel.to_numpy()
            # vel_mag = (vel[:, :, 0]**2.0+vel[:, :, 1]**2.0)**0.5
            # vel_img = cm.plasma(vel_mag / 0.15)
            
            ugrad = np.gradient(vel[:, :, 0])
            vgrad = np.gradient(vel[:, :, 1])
            vor = ugrad[1] - vgrad[0]
            # vor_img = abs(ugrad[1] - vgrad[0]) * 50
            # color map
            colors = [(1, 1, 0), (0.953, 0.490, 0.016), (0, 0, 0), (0.176, 0.976, 0.529), (0, 1, 1)]
            my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('my_cmap', colors)
            vor_img = cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=-0.02, vmax=0.02),cmap=my_cmap).to_rgba(vor)

            # img = np.concatenate((vor_img, vel_img), axis=1)
            
            gui.set_image(vor_img)

            # if i % 1000 == 0:
            #     video_manager.write_frame(gui.get_image())
            
            gui.show()

        # video_manager.make_video(gif=True, mp4=True)






lbm = LBM_Solver(   801,
                    201,
                    0.01,
                    [0, 1, 0, 0],
                    [[0.1, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                    [160.0, 100.0],
                    20.0,
                    500000
                )

lbm.solve()

