# # import scipy
import numpy as np
from numpy import linalg as la
import math
# P = np.array([[1,1],[1,-1]])
# # P = np.array([[5,5],[5,-5]]) # result doesn't change
# D = np.diag((3,1))
# M = P@D@la.inv(P)
# # print("P:\n", P)
# # print(la.inv(P))
# # print("D:\n", D)
# print(P@D) #dot
# print(P*D) #element-wise
# # print("M:\n", M)

# eigVals, eigVecs = la.eig(M)

# print("eigVals:\n", eigVals)
# print("eigVecs:\n", eigVecs)


# A = np.array([[1, 0],[1,-1]])

n = 20*3

def check_symmetric(A, rtol=1e-05, atol=1e-08):
    return np.allclose(A, A.T, rtol=rtol, atol=atol)

def check(A):

    print(A)

    eigVals, eigVecs = la.eig(A)
    print("eigenvalues:\n", eigVals)

    mx = 0
    for v in eigVals:
        mx = max(mx, np.absolute(v))
        print("abs(eigenvalue): ", np.absolute(v))

    print("Spectral radius:", mx)

    if check_symmetric(A): 
        print("This Matrix is symmetric.")

        mi = math.inf
        for v in eigVals:
            mi = min(mi, v)

        print("Minimum eigenvalue:", mi)

        if mi > 0: print("This Matrix is positive definite.")

    else: 
        print("This Matrix is **not** symmetric.")
        # print bad position (not symmetric)
        # for i in range(n):
        #     for j in range(n):
        #         if abs(A[i, j] - A[j, i]) > 1e-6:
        #             print("bad:", i, j, A[i, j], A[j, i])
    # for i in range(n):
    #     for j in range(i):
    #         if abs(A[i, j] - A[j, i]) > 1e-6:
    #             print("bad:", i, j, A[i, j], A[j, i])


def check_jacobi_iteration(A):
    S = np.zeros((n, n))
    T = np.zeros((n, n))
    for i in range(n):
        S[i, i] = A[i, i]
        for j in range(n):
            if i != j:
                T[i, j] = - A[i, j]
    B = la.inv(S)@T
    # print("B", B)
    # print("S", S)
    # print("T", T)

    # F = B
    # for i in range(1,5):
    #     C = B
    #     for j in range(i):
    #         C = C @ B
    #     F += C

    # I = np.identity(n)

    check(B)

def read_and_check(filename):
    A = np.zeros((n, n))
    with open(filename, "r") as file:
        for i in range(n):
            row = file.readline().split()
            for j in range(n):
                A[i, j] = float(row[j])
                # print(A[i, j])
    check(A)
    # check_jacobi_iteration(A)






# check("matrix_K_1e6.txt")
# check("matrix_A_1e6.txt")
# check("matrix_A_1e3.txt")
# check("matrix_K_1_1e3.txt")
# check("matrix_A_1_1e3.txt")
# check("matrix_K_1_1e6.txt")
# read_and_check("matrix_A_1_1e6.txt")
# read_and_check("matrix_A_1_1e3.txt")
# read_and_check("matrix_A_cg_1e3.txt")
# read_and_check("matrix_K_55_3.txt")
# read_and_check("matrixK_54.txt")
# read_and_check("matrix_A_cube_1e3.txt")
# read_and_check("matrix_A_L_1e3.txt")
read_and_check("matrix_A_L_1e3_1e-3.txt")
