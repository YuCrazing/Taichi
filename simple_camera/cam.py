import taichi as ti #version 0.6.10

@ti.data_oriented
class OBJ:
    def __init__(self, filename):

        self.v=[]
        self.f=[]

        # read data from *.obj file
        with open(filename, "r") as file:
            for line in file:
                if line[0:2] == 'v ':
                    for pos in line.strip().split()[1:4]: #[x, y, z]
                        self.v.append(float(pos))

                if line[0:2] == 'f ':
                    polygon = []
                    for all_ind in line.strip().split()[1:]:
                        v_ind = all_ind.strip().split('/')[0]
                        # print(v_ind)
                        polygon.append(int(v_ind)-1)
                    # print(polygon)
                    for i in range(1, len(polygon)-1, 1):
                        self.f += [polygon[0], polygon[i], polygon[i+1]]



        # print(self.v)
        # print(self.f)
        self.dt = 1e-3
        self.vn = int(len(self.v)/3)
        self.fn = int(len(self.f)/3)

        self.position = ti.Vector(3, dt=ti.f32, shape=self.vn)
        self.velocity = ti.Vector(3, dt=ti.f32, shape=self.vn)
        self.faces = ti.Vector(3, dt=ti.i32, shape=self.fn)

        self.ndc = ti.Vector(4, dt=ti.f32, shape=self.vn)


        self.cam_position = ti.Vector(3, dt=ti.f32, shape=())
        self.cam_target = ti.Vector(3, dt=ti.f32, shape=())
        self.cam_forward = ti.Vector(3, dt=ti.f32, shape=())
        self.cam_upDir = ti.Vector(3, dt=ti.f32, shape=())
        self.cam_left = ti.Vector(3, dt=ti.f32, shape=())
        self.cam_up = ti.Vector(3, dt=ti.f32, shape=())

        self.view_matrix = ti.Matrix(4, 4, dt=ti.f32, shape=())
        self.projection_matrix = ti.Matrix(4, 4, dt=ti.f32, shape=())

        self.proj_l = ti.var(dt=ti.f32, shape=())
        self.proj_r = ti.var(dt=ti.f32, shape=())
        self.proj_t = ti.var(dt=ti.f32, shape=())
        self.proj_b = ti.var(dt=ti.f32, shape=())
        self.proj_n = ti.var(dt=ti.f32, shape=())
        self.proj_f = ti.var(dt=ti.f32, shape=())


        print("vertices: ", self.vn, "triangles: ", self.fn)

        for i in range(self.vn):
            self.velocity[i] = [0, -10, 0]
            self.position[i] = [self.v[3*i], self.v[3*i+1], self.v[3*i+2]]

        for i in range(self.fn):
            self.faces[i] = [self.f[3*i], self.f[3*i+1], self.f[3*i+2]]


    @ti.kernel
    def initialize(self):
      
        # camera's target is the center of the object
        for i in range(self.vn):
            self.cam_target[None] += self.position[i]
        self.cam_target[None] /= self.vn
        self.cam_position[None].y = self.cam_target[None].y + 2
        self.cam_position[None].z = self.cam_target[None].z + 10

        self.proj_l[None] = -1
        self.proj_r[None] = 1
        self.proj_t[None] = 1
        self.proj_b[None] = -1
        self.proj_n[None] = 1
        self.proj_f[None] = 10000
        self.projection_matrix[None][0, 0] = 2*self.proj_n[None]/(self.proj_r[None]-self.proj_l[None])
        self.projection_matrix[None][0, 2] = (self.proj_r[None]+self.proj_l[None])/(self.proj_r[None]-self.proj_l[None])
        self.projection_matrix[None][1, 1] = 2*self.proj_n[None]/(self.proj_t[None]-self.proj_b[None])
        self.projection_matrix[None][1, 2] = (self.proj_t[None]+self.proj_b[None])/(self.proj_t[None]-self.proj_b[None])
        self.projection_matrix[None][2, 2] = -(self.proj_f[None]+self.proj_n[None])/(self.proj_f[None]-self.proj_n[None])
        self.projection_matrix[None][2, 3] = -2*self.proj_f[None]*self.proj_n[None]/(self.proj_f[None]-self.proj_n[None])
        self.projection_matrix[None][3, 2] = -1

    @ti.kernel
    def time_integrate(self):
        for i in range(self.vn):
            self.position[i] += self.velocity[i] * self.dt


    @ti.kernel
    def projection(self, frame:ti.i32):

        self.cam_forward[None] = self.cam_position[None] - self.cam_target[None]
        self.cam_forward[None] = self.cam_forward[None].normalized()
        self.cam_upDir[None] = ti.Vector([0, 1, 0])
        self.cam_left[None] = self.cam_upDir[None].cross(self.cam_forward[None])
        self.cam_up[None] = self.cam_forward[None].cross(self.cam_left[None])

        self.view_matrix[None][0, 0] = self.cam_left[None].x
        self.view_matrix[None][0, 1] = self.cam_left[None].y
        self.view_matrix[None][0, 2] = self.cam_left[None].z
        self.view_matrix[None][0, 3] = -self.cam_left[None].x * self.cam_position[None].x - self.cam_left[None].y * self.cam_position[None].y - self.cam_left[None].z * self.cam_position[None].z
        self.view_matrix[None][1, 0] = self.cam_up[None].x
        self.view_matrix[None][1, 1] = self.cam_up[None].y
        self.view_matrix[None][1, 2] = self.cam_up[None].z
        self.view_matrix[None][1, 3] = -self.cam_up[None].x * self.cam_position[None].x - self.cam_up[None].y * self.cam_position[None].y - self.cam_up[None].z * self.cam_position[None].z
        self.view_matrix[None][2, 0] = self.cam_forward[None].x
        self.view_matrix[None][2, 1] = self.cam_forward[None].y
        self.view_matrix[None][2, 2] = self.cam_forward[None].z
        self.view_matrix[None][2, 3] = -self.cam_forward[None].x * self.cam_position[None].x - self.cam_forward[None].y * self.cam_position[None].y - self.cam_forward[None].z * self.cam_position[None].z
        self.view_matrix[None][3, 3] = 1

        for i in range(self.vn):
            homogeneous_position = ti.Vector([self.position[i].x, self.position[i].y, self.position[i].z, 1])
            self.ndc[i] = self.projection_matrix[None] @ self.view_matrix[None] @ homogeneous_position
            self.ndc[i] /= self.ndc[i].w
            self.ndc[i].x += 1
            self.ndc[i].x /= 2
            self.ndc[i].y += 1
            self.ndc[i].y /= 2
            # print(homogeneous_position)
            # print(self.ndc[i])




ti.init(arch = ti.cpu)


obj = OBJ('box.obj')
# obj = OBJ('humanoid_quad.obj')
# obj = OBJ('dodecahedron.obj')
# obj = OBJ('Lowpoly_tree_sample.obj')
# obj = OBJ('teddy.obj')

obj.initialize()

win_width = 600
gui = ti.GUI('Simple Camera', res=(win_width, win_width))


for i in range(9999999):

    delta = 1e-1
    while gui.get_event(ti.GUI.PRESS):
        if gui.event.key == ti.GUI.ESCAPE:
            gui.running = False

    if gui.is_pressed(ti.GUI.LEFT, 'a'):
        obj.cam_position[None].x -= delta * obj.cam_left[None].x
        obj.cam_position[None].y -= delta * obj.cam_left[None].y
        obj.cam_position[None].z -= delta * obj.cam_left[None].z
        # print(obj.cam_position)
    if gui.is_pressed(ti.GUI.RIGHT, 'd'):
        obj.cam_position[None].x += delta * obj.cam_left[None].x
        obj.cam_position[None].y += delta * obj.cam_left[None].y
        obj.cam_position[None].z += delta * obj.cam_left[None].z
    if gui.is_pressed(ti.GUI.UP, 'w'):
        obj.cam_position[None].x -= delta * obj.cam_forward[None].x
        obj.cam_position[None].y -= delta * obj.cam_forward[None].y
        obj.cam_position[None].z -= delta * obj.cam_forward[None].z
    if gui.is_pressed(ti.GUI.DOWN, 's'):
        obj.cam_position[None].x += delta * obj.cam_forward[None].x
        obj.cam_position[None].y += delta * obj.cam_forward[None].y
        obj.cam_position[None].z += delta * obj.cam_forward[None].z
    if gui.is_pressed(ti.GUI.SPACE):
        obj.cam_position[None].x += delta * obj.cam_upDir[None].x
        obj.cam_position[None].y += delta * obj.cam_upDir[None].y
        obj.cam_position[None].z += delta * obj.cam_upDir[None].z
    if gui.is_pressed(ti.GUI.SHIFT):
        obj.cam_position[None].x -= delta * obj.cam_upDir[None].x
        obj.cam_position[None].y -= delta * obj.cam_upDir[None].y
        obj.cam_position[None].z -= delta * obj.cam_upDir[None].z


    # obj.time_integrate()

    obj.projection(i)

    for i in range(obj.fn):
        for j in range(3):
            # clip lines out of the frustum
            if abs(obj.ndc[obj.faces[i][j]].z) < 1 and abs(obj.ndc[obj.faces[i][(j+1)%3]].z) < 1:
                gui.line(obj.ndc[obj.faces[i][j]], obj.ndc[obj.faces[i][(j+1)%3]])
        # gui.triangle(obj.ndc[obj.faces[i][0]], obj.ndc[obj.faces[i][1]], obj.ndc[obj.faces[i][2]], color=0x000000+i*100)
    gui.show()