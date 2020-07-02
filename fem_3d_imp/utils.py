import taichi as ti

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
        self.position[None] = self.target[None] + ti.Vector([0, 0, 3.0*ti.sqrt((higherCorner[None]-lowerCorner[None]).dot(higherCorner[None]-lowerCorner[None]))])

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