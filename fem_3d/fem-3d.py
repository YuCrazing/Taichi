import taichi as ti #version 0.6.10
import math

@ti.data_oriented
class Camera:
    def __init__(self):

        self.position = ti.Vector(3, dt=ti.f32, shape=())
        self.target = ti.Vector(3, dt=ti.f32, shape=())
        self.forward = ti.Vector(3, dt=ti.f32, shape=())
        self.upDir = ti.Vector(3, dt=ti.f32, shape=())
        self.left = ti.Vector(3, dt=ti.f32, shape=())
        self.up = ti.Vector(3, dt=ti.f32, shape=())

        self.proj_l = -1
        self.proj_r = 1
        self.proj_t = 1
        self.proj_b = -1
        self.proj_n = 1
        self.proj_f = 10000

        self.view_matrix = ti.Matrix(4, 4, dt=ti.f32, shape=())
        self.projection_matrix = ti.Matrix(4, 4, dt=ti.f32, shape=())

    @ti.kernel
    def initialize(self, center:ti.template(), lowerCorner:ti.template(), higherCorner:ti.template()):

        self.target[None] = center[None]
        self.position[None] = self.target[None] + ti.Vector([0, 0, 2.5*ti.sqrt((higherCorner[None]-lowerCorner[None]).dot(higherCorner[None]-lowerCorner[None]))])

        self.projection_matrix[None][0, 0] = 2*self.proj_n/(self.proj_r-self.proj_l)
        self.projection_matrix[None][0, 2] = (self.proj_r+self.proj_l)/(self.proj_r-self.proj_l)
        self.projection_matrix[None][1, 1] = 2*self.proj_n/(self.proj_t-self.proj_b)
        self.projection_matrix[None][1, 2] = (self.proj_t+self.proj_b)/(self.proj_t-self.proj_b)
        self.projection_matrix[None][2, 2] = -(self.proj_f+self.proj_n)/(self.proj_f-self.proj_n)
        self.projection_matrix[None][2, 3] = -2*self.proj_f*self.proj_n/(self.proj_f-self.proj_n)
        self.projection_matrix[None][3, 2] = -1

    @ti.kernel
    def calculate_view_matrix(self):

        self.forward[None] = self.position[None] - self.target[None]
        self.forward[None] = self.forward[None].normalized()
        self.upDir[None] = ti.Vector([0, 1, 0])
        self.left[None] = self.upDir[None].cross(self.forward[None])
        self.up[None] = self.forward[None].cross(self.left[None])

        self.view_matrix[None][0, 0] = self.left[None].x
        self.view_matrix[None][0, 1] = self.left[None].y
        self.view_matrix[None][0, 2] = self.left[None].z
        self.view_matrix[None][0, 3] = -self.left[None].x * self.position[None].x - self.left[None].y * self.position[None].y - self.left[None].z * self.position[None].z
        self.view_matrix[None][1, 0] = self.up[None].x
        self.view_matrix[None][1, 1] = self.up[None].y
        self.view_matrix[None][1, 2] = self.up[None].z
        self.view_matrix[None][1, 3] = -self.up[None].x * self.position[None].x - self.up[None].y * self.position[None].y - self.up[None].z * self.position[None].z
        self.view_matrix[None][2, 0] = self.forward[None].x
        self.view_matrix[None][2, 1] = self.forward[None].y
        self.view_matrix[None][2, 2] = self.forward[None].z
        self.view_matrix[None][2, 3] = -self.forward[None].x * self.position[None].x - self.forward[None].y * self.position[None].y - self.forward[None].z * self.position[None].z
        self.view_matrix[None][3, 3] = 1

@ti.data_oriented
class Floor:
    def __init__(self, height, scale):
        self.position = ti.Vector(3, dt=ti.f32, shape=4)
        self.ndc = ti.Vector(4, dt=ti.f32, shape=4)
        self.height = height
        self.scale = scale

    @ti.kernel
    def initialize(self):
        h = self.height
        k = self.scale
        self.position[0] = [-k, h, -k]
        self.position[1] = [k, h, -k]
        self.position[2] = [k, h, k]
        self.position[3] = [-k, h, k]

    @ti.kernel
    def projection(self, view_matrix:ti.template(), projection_matrix:ti.template()):

        for i in range(4):
            homogeneous_position = ti.Vector([self.position[i].x, self.position[i].y, self.position[i].z, 1])
            self.ndc[i] = projection_matrix[None] @ view_matrix[None] @ homogeneous_position
            self.ndc[i] /= self.ndc[i].w
            self.ndc[i].x += 1
            self.ndc[i].x /= 2
            self.ndc[i].y += 1
            self.ndc[i].y /= 2

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

        self.node = ti.Vector(self.dim, dt=ti.f32, shape=self.vn, needs_grad=True)
        self.face = ti.Vector(3, dt=ti.i32, shape=self.fn)
        self.element = ti.Vector(4, dt=ti.i32, shape=self.en)

        self.ndc = ti.Vector(self.dim+1, dt=ti.f32, shape=self.vn)

        self.center = ti.Vector(self.dim, ti.f32, shape=())
        self.lowerCorner = ti.Vector(self.dim, ti.f32, shape=())
        self.higherCorner = ti.Vector(self.dim, ti.f32, shape=())

        ## for simulation
        self.E = 1000 # Young modulus
        self.nu = 0.3 # Poisson's ratio: nu \in [0, 0.5)
        self.mu = self.E / (2 * (1 + self.nu))
        self.la = self.E * self.nu / ((1 + self.nu) * (1 -2 * self.nu))
        self.dt = 5e-4
        self.velocity = ti.Vector(self.dim, dt=ti.f32, shape=self.vn)
        self.neighbor_element_count = ti.var(dt=ti.i32, shape=self.vn)
        self.node_mass = ti.var(dt=ti.f32, shape=self.vn)
        self.element_mass = ti.var(dt=ti.f32, shape=self.en)
        self.element_volume = ti.var(dt=ti.f32, shape=self.en)
        self.B = ti.Matrix(self.dim, self.dim, dt=ti.f32, shape=self.en) # a square matrix
        self.energy = ti.var(dt=ti.f32, shape=(), needs_grad=True)

        # for rendering
        self.color = ti.var(dt=ti.i32, shape=self.vn)


        print("vertices: ", self.vn, "elements: ", self.en)

        for i in range(self.vn):
            self.node[i] = [self.v[3*i], self.v[3*i+1], self.v[3*i+2]]
            self.velocity[i] = [0, -10, 0]

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
            self.element_volume[i] = abs(D.determinant()) / 6
            self.element_mass[i] = self.element_volume[i]
            self.node_mass[a] += self.element_mass[i]
            self.node_mass[b] += self.element_mass[i]
            self.node_mass[c] += self.element_mass[i]
            self.node_mass[d] += self.element_mass[i]
            self.neighbor_element_count[a] += 1
            self.neighbor_element_count[b] += 1
            self.neighbor_element_count[c] += 1
            self.neighbor_element_count[d] += 1
            # print(i, "element_mass", self.element_mass[i], "element_volume", self.element_volume[i])

        for i in range(self.vn):
            self.node_mass[i] /= max(self.neighbor_element_count[i], 1)
            # print(i, "node_mass", self.node_mass[i])


    @ti.kernel
    def projection(self, view_matrix:ti.template(), projection_matrix:ti.template()):

        for i in range(self.vn):
            homogeneous_position = ti.Vector([self.node[i].x, self.node[i].y, self.node[i].z, 1])
            self.ndc[i] = projection_matrix[None] @ view_matrix[None] @ homogeneous_position
            self.ndc[i] /= self.ndc[i].w
            self.ndc[i].x += 1
            self.ndc[i].x /= 2
            self.ndc[i].y += 1
            self.ndc[i].y /= 2


    @ti.func
    def D(self, i):
        a = self.element[i][0]
        b = self.element[i][1]
        c = self.element[i][2]
        d = self.element[i][3]
        return ti.Matrix.cols([self.node[b]-self.node[a], self.node[c]-self.node[a], self.node[d]-self.node[a]])

    @ti.func
    def F(self, i): # deformation gradient
        return self.D(i) @ self.B[i]

    @ti.func
    def Psi(self, i): # (strain) energy density
        F = self.F(i)
        J = max(F.determinant(), 0.01)
        return self.mu / 2 * ( (F @ F.transpose()).trace() - self.dim ) - self.mu * ti.log(J) + self.la / 2 * ti.log(J)**2

    @ti.func
    def U0(self, i): # elastic potential energy
        return self.element_volume[i] * self.Psi(i)

    @ti.func
    def U1(self, i): # gravitational potential energy
        a = self.element[i][0]
        b = self.element[i][1]
        c = self.element[i][2]
        d = self.element[i][3]
        return self.element_mass[i] * 10 * 4 * (self.node[a].y + self.node[b].y + self.node[c].y + self.node[d].y) / 4

    @ti.kernel
    def energy_integrate(self):
        for i in range(self.en):
            self.energy[None] += self.U0(i) + self.U1(i)

    @ti.kernel
    def time_integrate(self, floor_height:ti.f32):
        
        mx = -self.inf

        for i in range(self.vn):
            self.velocity[i] += ( - self.node.grad[i] / self.node_mass[i]  ) * self.dt
            self.velocity[i] *= math.exp(self.dt*-6)
            self.node[i] += self.velocity[i] * self.dt

            if self.node[i].y < floor_height:
                self.node[i].y = floor_height
                self.velocity[i].y = 0

            mx = max(mx, self.node.grad[i].norm())

        mx = max(mx, 1)

        for i in range(self.vn):
            self.color[i] = int(self.node.grad[i].norm() / mx * 255)







ti.init(arch = ti.cpu)



cam = Camera()
floor = Floor(-2, 4)
# obj = Object('tetrahedral-models/cube.1')
obj = Object('tetrahedral-models/ellell.1', 0)

obj.initialize()
floor.initialize()
cam.initialize(obj.center, obj.lowerCorner, obj.higherCorner)

win_width = 600
gui = ti.GUI('Finite Element Method', res=(win_width, win_width), background_color=0x112F41)

get_color = lambda i, j1, j2: int((obj.color[obj.element[i][j1]] + obj.color[obj.element[i][j2]])/2)*0x010000

for i in range(9999999):

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

    cam.calculate_view_matrix()


    floor.projection(cam.view_matrix, cam.projection_matrix)


    for i in range(10):
        with ti.Tape(obj.energy):
            obj.energy_integrate()
        obj.time_integrate(floor.height)

    obj.projection(cam.view_matrix, cam.projection_matrix)


    # render floor
    for i in range(4):
        if abs(floor.ndc[i].z) < 1 and abs(floor.ndc[(i+1)%4].z) < 1:
            gui.line(floor.ndc[i], floor.ndc[(i+1)%4], color=0x3241f4)
    # for i in range(0, 3, 2):
    #     if abs(floor.ndc[i].z) < 1 and abs(floor.ndc[(i+1)%4].z) and abs(floor.ndc[(i+2)%4].z) < 1:
    #         gui.triangle(floor.ndc[i], floor.ndc[(i+2)%4], floor.ndc[(i+1)%4], color=0x3241f4)


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

    gui.show()