import numpy as np
import scipy
from scipy.sparse.linalg import factorized
from scipy.sparse import dia_matrix, csr_matrix

class MultiGridSolver:
  def __init__(self, A, Is, w=2/3):
    """
    Inputs:
        A   sparse matrix
        Is  list of tuples (I_f2c, I_c2f), where I_f2c is restriction matrix from
            fine grid to coarse grid, I_c2f is interploation matrix from coarse
            grid to fine grid
        w   parameter for weighted Jacobian method, default 2/3
    """
    self.A = A
    self.Is = Is
    self.size = A.shape[0]
    self.w = w

    # build multi-level weighted Jacobian Rw = I - w D^{-1} A
    self.As = []
    self.Rws = []
    self.D_invs=[]

    A_tmp = A
    for I_f2c, I_c2f in Is:
      self.As.append(A_tmp)

      # build weighted Jacobian
      D = A_tmp.diagonal()
      m, n = A_tmp.shape
      D_inv = scipy.sparse.dia_matrix((1/D, 0), shape=(m, n))
      Rw = scipy.sparse.eye(m) - w * D_inv * A_tmp
      self.Rws.append(Rw)
      self.D_invs.append(1/D)

      # go down to coarse grid
      A_tmp = I_f2c * A_tmp * I_c2f

    # build solver for coarsest grid
    self.coarsest_solver = factorized(A_tmp)

  def solve(self, f, v_init=None, alpha=1, eps=1e-6):
    if v_init is None or v.size() != self.size:
      v = np.zeros(self.size)
    else:
      v = v_init

    res = f - self.A * v

    err = np.linalg.norm(res)
    print("start err %f" % err)

    cnt = 0
    while err > eps:
      cnt+=1
      v_add = self._step(res, 0, alpha)
      v += v_add
      res = f - self.A * v
      err = np.linalg.norm(res)
      print("iter %d, err = %f" % (cnt, err))

    return v

  def _step(self, res, depth, alpha=1):
    if depth == len(self.Rws):
      v_out = self.coarsest_solver(res)
      return v_out

    # relax
    v_relax = np.zeros_like(res)
    for i in range(alpha):
      v_relax = self.Rws[depth] * v_relax + (1-self.w) * res * self.D_invs[depth]

    # to coarse grid and back
    res_relax = res - self.As[depth] * v_relax
    r_coarse = self.Is[depth][0] * res_relax
    v_coarse = self._step(r_coarse, depth+1)
    v_relax = v_relax + self.Is[depth][1] * v_coarse

    # relax again
    for i in range(alpha):
      v_relax = self.Rws[depth] * v_relax + (1-self.w) * res * self.D_invs[depth]

    return v_relax

def poisson_1d(n, depth):
  size = n
  diag = np.ones(n)
  A = dia_matrix(([-diag, 2*diag, -diag], [-1, 0, 1]), shape=(n,n))

  Is = []
  for i in range(depth):
    if n < 100:  # coarse grid not need to be too small
      break

    l = n // 2
    I_tmp = np.zeros((l, 2*l+1))
    for j in range(l):
      I_tmp[j, 2*j] = 1
      I_tmp[j, 2*j+1] = 2
      I_tmp[j, 2*j+2] = 1
    I_f2c = I_tmp[:, :n] / 4
    I_c2f = I_f2c.transpose() * 2

    I_f2c = scipy.sparse.csr_matrix(I_f2c)
    I_c2f = scipy.sparse.csr_matrix(I_c2f)

    Is.append((I_f2c, I_c2f))

    n = n // 2

  multi_solver = MultiGridSolver(A, Is)

  rng = np.random.RandomState(1124)
  f = rng.rand(size)
  f = f - f.mean()
  v = multi_solver.solve(f * 1/size, alpha=1)

  # plot the solution
  import matplotlib.pyplot as plt
  plt.plot(v)
  plt.plot(f)
  plt.show()

if __name__=="__main__":
  poisson_1d(1024, 10)