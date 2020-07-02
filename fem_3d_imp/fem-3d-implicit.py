import taichi as ti #version 0.6.10
import numpy as np
import math

from utils import *

@ti.data_oriented
class Object:
    def __init__(self, filename, index_start=1):

        self.v = []
        self.f = []
        self.e = []

        # read nodes from *.node file
        with open(filename+".node", "r") as file:
            vn = int(file.readline().split()[0])
            for i in range(vn):
                self.v += [ float(x) for x in file.readline().split()[1:4] ] #[x , y, z]

        # read faces from *.face file (only for rendering)
        with open(filename+".face", "r") as file:
            fn = int(file.readline().split()[0])
            for i in range(fn):
                self.f += [ int(ind)-index_start for ind in file.readline().split()[1:4] ] # triangle

        # read elements from *.ele file
        with open(filename+".ele", "r") as file:
            en = int(file.readline().split()[0])
            for i in range(en): # !!!!! some files are 1-based indexed
                self.e += [ int(ind)-index_start for ind in file.readline().split()[1:5] ] # tetrahedron



        # print(self.v)
        # print(self.f)
        # print(self.e)
        self.vn = int(len(self.v)/3)
        self.fn = int(len(self.f)/3)
        self.en = int(len(self.e)/4)
        self.dim = 3
        self.inf = 1e10

        self.node = ti.Vector(self.dim, dt=ti.f32, shape=self.vn)
        self.face = ti.Vector(3, dt=ti.i32, shape=self.fn)
        self.element = ti.Vector(4, dt=ti.i32, shape=self.en)

        self.ndc = ti.Vector(self.dim+1, dt=ti.f32, shape=self.vn)

        self.center = ti.Vector(self.dim, ti.f32, shape=())
        self.lowerCorner = ti.Vector(self.dim, ti.f32, shape=())
        self.higherCorner = ti.Vector(self.dim, ti.f32, shape=())


        # 0: explicit time integration
        # 1: jacobi iteration
        # 2: conjugate gradient
        self.method = 0 

        ## for simulation
        self.dt = (1.0/24)
        self.g = 10.0
        self.E = 1e3 # Young's modulus # system will explode when 1e6
        self.nu = 0.3 # Poisson's ratio: nu \in [0, 0.5)
        # self.mu = self.E / (2 * (1 + self.nu))
        # self.la = self.E * self.nu / ((1 + self.nu) * (1 -2 * self.nu))
        self.mu = ti.var(dt=ti.f32, shape=())
        self.la = ti.var(dt=ti.f32, shape=())
        self.velocity = ti.Vector(self.dim, dt=ti.f32, shape=self.vn)
        self.force = ti.Vector(self.dim, dt=ti.f32, shape=self.vn)
        self.node_mass = ti.var(dt=ti.f32, shape=self.vn)
        self.element_volume = ti.var(dt=ti.f32, shape=self.en)
        self.B = ti.Matrix(self.dim, self.dim, dt=ti.f32, shape=self.en) # a square matrix

        # derivatives
        self.D_ndim = ti.Matrix(self.dim, self.dim, dt=ti.f32, shape=(self.en, 4, 3))
        self.F_ndim = ti.Matrix(self.dim, self.dim, dt=ti.f32, shape=(self.en, 4, 3))
        self.P_ndim = ti.Matrix(self.dim, self.dim, dt=ti.f32, shape=(self.en, 4, 3))
        self.H_ndim = ti.Matrix(self.dim, self.dim, dt=ti.f32, shape=(self.en, 4, 3))
        self.K = ti.var(dt=ti.f32, shape=(self.dim*self.vn, self.dim*self.vn))

        # for solving system of linear equations
        self.A = ti.var(dt=ti.f32, shape=(self.dim*self.vn, self.dim*self.vn))
        self.b = ti.var(dt=ti.f32, shape=self.dim*self.vn)
        self.x = ti.var(dt=ti.f32, shape=self.dim*self.vn)
        self.x_new = ti.var(dt=ti.f32, shape=self.dim*self.vn)
        self.r = ti.var(dt=ti.f32, shape=self.dim*self.vn) # residual
        self.d = ti.var(dt=ti.f32, shape=self.dim*self.vn)
        self.q = ti.var(dt=ti.f32, shape=self.dim*self.vn)

        # for rendering
        self.color = ti.var(dt=ti.i32, shape=self.vn)


        print("vertices: ", self.vn, "elements: ", self.en)

        for i in range(self.vn):
            self.node[i] = [self.v[3*i], self.v[3*i+1], self.v[3*i+2]]
            self.velocity[i] = [0, 0, 0]

        for i in range(self.fn):
            self.face[i] = [self.f[3*i], self.f[3*i+1], self.f[3*i+2]]

        for i in range(self.en):
            self.element[i] = [self.e[4*i], self.e[4*i+1], self.e[4*i+2], self.e[4*i+3]]

        # calculate the center of the object (the target of the camera)
        self.lowerCorner[None] = [self.inf, self.inf, self.inf]
        self.higherCorner[None] = [-self.inf, -self.inf, -self.inf]
        for i in range(self.vn):
            self.center[None].x += self.node[i].x
            self.center[None].y += self.node[i].y
            self.center[None].z += self.node[i].z
            self.lowerCorner[None].x = min(self.lowerCorner[None].x, self.node[i].x)
            self.lowerCorner[None].y = min(self.lowerCorner[None].y, self.node[i].y)
            self.lowerCorner[None].z = min(self.lowerCorner[None].z, self.node[i].z)
            self.higherCorner[None].x = max(self.higherCorner[None].x, self.node[i].x)
            self.higherCorner[None].y = max(self.higherCorner[None].y, self.node[i].y)
            self.higherCorner[None].z = max(self.higherCorner[None].z, self.node[i].z)

        self.center[None].x /= max(self.vn, 1)
        self.center[None].y /= max(self.vn, 1)
        self.center[None].z /= max(self.vn, 1)



    @ti.kernel
    def initialize(self):

        for i in range(self.en):
            D = self.D(i)
            self.B[i] = D.inverse()
            a, b, c, d = self.element[i][0], self.element[i][1], self.element[i][2], self.element[i][3]
            self.element_volume[i] = abs(D.determinant()) / 6.0


        for i in range(self.vn):
            self.node_mass[i] = 1.0

    @ti.kernel
    def projection(self, view_matrix:ti.template(), projection_matrix:ti.template()):

        for i in range(self.vn):
            homogeneous_position = ti.Vector([self.node[i].x, self.node[i].y, self.node[i].z, 1])
            self.ndc[i] = projection_matrix[None] @ view_matrix[None] @ homogeneous_position
            self.ndc[i] /= self.ndc[i].w
            self.ndc[i].x += 1.0
            self.ndc[i].x /= 2.0
            self.ndc[i].y += 1.0
            self.ndc[i].y /= 2.0

    @ti.func
    def D(self, i):
        a = self.element[i][0]
        b = self.element[i][1]
        c = self.element[i][2]
        d = self.element[i][3]
        return ti.Matrix.cols([self.node[a]-self.node[d], self.node[b]-self.node[d], self.node[c]-self.node[d]])

    @ti.func
    def F(self, i): # deformation gradient
        return self.D(i) @ self.B[i]

    @ti.func
    def P(self, i):
        F = self.F(i)
        F_T = F.inverse().transpose()
        J = max(F.determinant(), 0.01)
        return self.mu * (F - F_T) + self.la * ti.log(J) * F_T

    @ti.kernel
    def compute_force(self):

        for i in range(self.vn):
            self.force[i] = ti.Vector([0, -self.node_mass[i] * self.g, 0])

        for i in range(self.en):

            P = self.P(i)

            H = - self.element_volume[i] * (P @ self.B[i].transpose())

            h1 = ti.Vector([H[0, 0], H[1, 0], H[2, 0]])
            h2 = ti.Vector([H[0, 1], H[1, 1], H[2, 1]])
            h3 = ti.Vector([H[0, 2], H[1, 2], H[2, 2]])

            a = self.element[i][0]
            b = self.element[i][1]
            c = self.element[i][2]
            d = self.element[i][3]


            self.force[a] += h1
            self.force[b] += h2
            self.force[c] += h3
            self.force[d] += - (h1 + h2 + h3)


    @ti.func
    def compute_K(self):
        for e in range(self.en):
            for n in range(4):
                for dim in range(self.dim):
                    for i in ti.static(range(self.dim)):
                        for j in ti.static(range(self.dim)):
                            self.D_ndim[e, n, dim][i, j] = 0
                            self.F_ndim[e, n, dim][i, j] = 0
                            self.P_ndim[e, n, dim][i, j] = 0

            for n in ti.static(range(3)):
                for dim in ti.static(range(self.dim)):
                    self.D_ndim[e, n, dim][dim, n] = 1

            for dim in ti.static(range(self.dim)):
                self.D_ndim[e, 3, dim] = - (self.D_ndim[e, 0, dim] + self.D_ndim[e, 1, dim] + self.D_ndim[e, 2, dim])

            for n in range(4):
                for dim in range(self.dim):
                    self.F_ndim[e, n, dim] = self.D_ndim[e, n, dim] @ self.B[e] # !!!!! matrix multiplication

            F = self.F(e)
            F_1 = F.inverse()
            F_1_T = F_1.transpose()
            J = max(F.determinant(), 0.01)

            for n in range(4):
                for dim in range(self.dim):
                    for i in ti.static(range(self.dim)):
                        for j in ti.static(range(self.dim)):
                            
                            # dF/dF_{ij}
                            dF = ti.Matrix([[0.0, 0.0, 0.0], 
                                            [0.0, 0.0, 0.0],
                                            [0.0, 0.0, 0.0]])
                            dF[i, j] = 1

                            # dF^T/dF_{ij}
                            dF_T = dF.transpose()

                            # Tr( F^{-1} dF/dF_{ij} )
                            dTr = F_1_T[i, j]

                            dP_dFij = self.mu * dF + (self.mu - self.la * ti.log(J)) * F_1_T @ dF_T @ F_1_T + self.la * dTr * F_1_T
                            dFij_ndim = self.F_ndim[e, n, dim][i, j]

                            self.P_ndim[e, n, dim] += dP_dFij * dFij_ndim

            for n in range(4):
                for dim in range(self.dim):
                    self.H_ndim[e, n, dim] = - self.element_volume[e] * self.P_ndim[e, n, dim] @ self.B[e].transpose()


    
            for n in ti.static(range(4)):
                i = self.element[e][n]
                for dim in ti.static(range(self.dim)):
                    ind = i * self.dim + dim
                    for j in ti.static(range(3)):
                        self.K[self.element[e][j]*3+0, ind] += self.H_ndim[e, n, dim][0, j] # df_jx/d_node_ndim
                        self.K[self.element[e][j]*3+1, ind] += self.H_ndim[e, n, dim][1, j] # df_jy/d_node_ndim
                        self.K[self.element[e][j]*3+2, ind] += self.H_ndim[e, n, dim][2, j] # df_jz/d_node_ndim

                    # df_3x/d_node_ndim
                    self.K[self.element[e][3]*3+0, ind] += -(self.H_ndim[e, n, dim][0, 0] + self.H_ndim[e, n, dim][0, 1] + self.H_ndim[e, n, dim][0, 2])
                    # df_3y/d_node_ndim
                    self.K[self.element[e][3]*3+1, ind] += -(self.H_ndim[e, n, dim][1, 0] + self.H_ndim[e, n, dim][1, 1] + self.H_ndim[e, n, dim][1, 2])
                    # df_3x/d_node_ndim
                    self.K[self.element[e][3]*3+2, ind] += -(self.H_ndim[e, n, dim][2, 0] + self.H_ndim[e, n, dim][2, 1] + self.H_ndim[e, n, dim][2, 2])

    @ti.kernel
    def assembly(self):

        for i, j in self.A:
            self.K[i, j] = 0
            self.A[i, j] = 0

        self.compute_K()

        for i in range(self.vn):
            for j in range(self.dim):
                for k in range(self.vn*self.dim):
                        self.K[i*self.dim+j, k] *= self.dt**2 / self.node_mass[i] 

        for i in range(self.vn):
            for j in range(self.dim):
               self.A[i*self.dim+j, i*self.dim+j] = 1

        for i, j in self.A:
            self.A[i, j] -= self.K[i, j]

            

        for i in range(self.vn):
            for j in ti.static(range(self.dim)):
                self.x[i*self.dim+j] = self.velocity[i][j] # initial values
                self.b[i*self.dim+j] = self.velocity[i][j] + self.dt / self.node_mass[i] * self.force[i][j]

    @ti.kernel
    def time_integrate(self, floor_height:ti.f32):

        # print(self.E) # E is still 1000 even though E is changed in keyboard events

        mx = -self.inf

        for i in range(self.vn):

            if self.method == 0:
                self.velocity[i] += ( self.force[i] / self.node_mass[i] + ti.Vector([0, -10, 0])  ) * self.dt
            else:
                self.velocity[i] = ti.Vector([self.x[i*3+0], self.x[i*3+1], self.x[i*3+2]])

            # self.velocity[i] *= math.exp(self.dt*-6)
            self.node[i] += self.velocity[i] * self.dt

            if self.node[i].y < floor_height:
                self.node[i].y = floor_height
                self.velocity[i].y = 0
                self.x[i*3+1] = 0

            mx = max(mx, self.force[i].norm())

        mx = max(mx, 1)

        for i in range(self.vn):
            self.color[i] = int(self.force[i].norm() / mx * 255)


    @ti.kernel
    def jacobi(self, max_iter_num:ti.i32, tol:ti.f32): # Jacobi iteration
        n = self.vn * self.dim
        iter_i = 0
        res = 0.0
        while iter_i < max_iter_num:

            for i in range(n): # every row
                r = self.b[i]*1.0
                for j in range(n): # every column
                    if i != j:
                        r -= self.A[i, j] * self.x[j]
                self.x_new[i] = r / self.A[i, i]

            for i in range(n):
                self.x[i] = self.x_new[i]

            res = 0.0 #!!!!!!!!!!!!!! residual = 0 will forever be an integer
            for i in range(n):
                r = self.b[i]*1.0
                for j in range(n):
                    r -= self.A[i, j] * self.x[j]
                res += r*r

            if res < tol:
                break

            iter_i += 1
        print("Jacobi iteration:", iter_i, res)




    @ti.kernel
    def CG(self, max_iter_num:ti.i32, tol:ti.f32): # conjugate gradient
        
        n = self.vn * self.dim
        
        for i in range(n): 
            r = self.b[i]*1.0
            for j in range(n):
                r -= self.A[i, j] * self.x[j]
            self.r[i] = r
            self.d[i] = r
        
        delta_new = 0.0
        for i in range(n):
            delta_new += self.r[i]*self.r[i]
        delta_0 = delta_new

        # tol = tol*tol

        iter_i = 0
        while iter_i < max_iter_num and delta_new > tol * delta_0:
            
            for i in range(n):
                r = 0.0 #!!!!!!!!!!!!!!!!! 0.0 not 0 !!!!!!
                for j in range(n):
                    r += self.A[i, j] * self.d[j]
                self.q[i] = r
            
            alpha = 0.0
            for i in range(n):
                alpha += self.d[i] * self.q[i]
            alpha = delta_new / alpha

            for i in range(n):
                self.x[i] += alpha * self.d[i]

            if (iter_i+1) % 1 == 0:
                for i in range(n):
                    r = self.b[i]*1.0
                    for j in range(n):
                        r -= self.A[i, j] * self.x[j]
                    self.r[i] = r
            else:
                for i in range(n):
                    self.r[i] -= alpha * self.q[i]

            delta_old = delta_new
            delta_new = 0.0
            for i in range(n):
                delta_new += self.r[i] * self.r[i]

            # if delta_new < tol: break

            beta = delta_new / delta_old

            for i in range(n):
                self.d[i] = self.r[i] + beta * self.d[i]

            iter_i += 1

        # res = 0.0 #!!!!!!!!!!!!!! residual = 0 will forever be an integer
        # for i in range(n):
        #     res += self.r[i] * self.r[i]

        print("Conjugate gradient:", iter_i, delta_new)



ti.init(arch=ti.cpu)
# ti.init(arch=ti.cpu, debug=True)



cam = Camera()
floor = Floor(-2, 4)
# obj = Object('tetrahedral-models/cube.1')
obj = Object('tetrahedral-models/ellell.1', 0)

obj.method = 2
obj.E = 1e6
obj.g = 10.0
calculation_time = (1 if obj.method == 2 else 30)
obj.dt = 1.0/24/calculation_time



obj.initialize()
floor.initialize()
cam.initialize(obj.center, obj.lowerCorner, obj.higherCorner)

win_width = 600
gui = ti.GUI('Implicit FEM', res=(win_width, win_width), background_color=0x112F41)

get_color = lambda i, j1, j2: int((obj.color[obj.element[i][j1]] + obj.color[obj.element[i][j2]])/2)*0x010000


for frame in range(9999999):

    delta = 2e-1
    while gui.get_event(ti.GUI.PRESS):
        if gui.event.key == ti.GUI.ESCAPE:
            gui.running = False

    if gui.is_pressed('a'):
        for i in range(3):
            cam.position[None][i] -= delta * cam.left[None][i]
    if gui.is_pressed('d'):
        for i in range(3):
            cam.position[None][i] += delta * cam.left[None][i]
    if gui.is_pressed('w'):
        for i in range(3):
            cam.position[None][i] -= delta * cam.forward[None][i]
    if gui.is_pressed('s'):
        for i in range(3):
            cam.position[None][i] += delta * cam.forward[None][i]
    if gui.is_pressed(ti.GUI.SPACE):
        for i in range(3):
            cam.position[None][i] += delta * cam.upDir[None][i]
    if gui.is_pressed(ti.GUI.SHIFT):
        for i in range(3):
            cam.position[None][i] -= delta * cam.upDir[None][i]

    if gui.is_pressed(ti.GUI.RETURN):
        for i in range(obj.vn):
            obj.velocity[i].y += 3

    if gui.is_pressed(ti.GUI.UP):
        obj.E *= 1.1

    if gui.is_pressed(ti.GUI.DOWN):
        obj.E /= 1.1


    cam.calculate_view_matrix()

    floor.projection(cam.view_matrix, cam.projection_matrix)


    obj.mu[None] = obj.E / (2 * (1 + obj.nu))
    obj.la[None] = obj.E * obj.nu / ((1 + obj.nu) * (1 - 2 * obj.nu))


    

    for i in range(calculation_time):
        obj.compute_force()

        if obj.method != 0:
            obj.assembly()

            # if frame == 0:
            #     with open("matrix_A_L_1e3_1e-3.txt", "w") as file:
            #         for i in range(obj.vn*obj.dim):
            #             for j in range(obj.vn*obj.dim):
            #                 file.write("%15.6f"%obj.A[i, j])
            #             file.write("\n")

            if obj.method == 1:
                # jacobi iteration
                obj.jacobi(100, 1e-6)
            elif obj.method == 2:
                # conjugate gradient
                obj.CG(obj.vn*obj.dim*3, 1e-4)
        
        obj.time_integrate(floor.height)
    # break


    obj.projection(cam.view_matrix, cam.projection_matrix)


    # render floor
    for i in range(4):
        if abs(floor.ndc[i].z) < 1 and abs(floor.ndc[(i+1)%4].z) < 1:
            gui.line(floor.ndc[i], floor.ndc[(i+1)%4], color=0x3241f4)


    # render object (faces)
    for i in range(obj.fn):
        for j in range(3):
            if abs(obj.ndc[obj.face[i][j]].z) < 1 and abs(obj.ndc[obj.face[i][(j+1)%3]].z) < 1:
                gui.line(obj.ndc[obj.face[i][j]], obj.ndc[obj.face[i][(j+1)%3]], color=get_color(i, j, (j+1)%3))

    # # render object (elements)
    # for i in range(obj.en):
    #     for j in range(4):
    #         if abs(obj.ndc[obj.element[i][j]].z) < 1 and abs(obj.ndc[obj.element[i][(j+1)%4]].z) < 1:
    #             gui.line(obj.ndc[obj.element[i][j]], obj.ndc[obj.element[i][(j+1)%4]], color=get_color(i, j, (j+1)%3))
    #     for j in range(0, 2):
    #         if abs(obj.ndc[obj.element[i][j]].z) < 1 and abs(obj.ndc[obj.element[i][j+2]].z) < 1:
    #             gui.line(obj.ndc[obj.element[i][j]], obj.ndc[obj.element[i][j+2]], color=get_color(i, j, j+2))

    gui.text(content=f'Nodes: {obj.vn}', pos=(0, 0.9), color=0xFFFFFF)
    gui.text(content=f'Elements: {obj.en}', pos=(0, 0.87), color=0xFFFFFF)
    gui.text(content=f'Young\'s modulus: {int(obj.E)}', pos=(0, 0.84), color=0xFFFFFF)

    filename = f'out-figs-normal-{obj.method}/frame_{frame:05d}.png'
    gui.show()
    # gui.show(filename)

    # export to .PLY file
    # if frame == 0:
    #     np_pos = np.reshape(floor.position.to_numpy(), (4, 3))
    #     np_face = np.array([0, 1, 2, 2, 3, 0])
    #     floor_writer = ti.PLYWriter(num_vertices=4, num_faces=2)
    #     floor_writer.add_vertex_pos(np_pos[:, 0], np_pos[:, 1], np_pos[:, 2])
    #     floor_writer.add_faces(np_face)
    #     floor_writer.export_frame_ascii(frame, "output/floor")

    # np_pos = np.reshape(obj.node.to_numpy(), (obj.vn, 3))
    # np_face = np.reshape(obj.face.to_numpy(), (1, -1))
    # obj_writer = ti.PLYWriter(num_vertices=obj.vn, num_faces=obj.fn)
    # obj_writer.add_vertex_pos(np_pos[:, 0], np_pos[:, 1], np_pos[:, 2])
    # obj_writer.add_faces(np_face)
    # obj_writer.export_frame_ascii(frame, "output-mesh-0/obj")
    
    # if frame+1 == 24*5: break