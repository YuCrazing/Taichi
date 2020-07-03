import numpy as np
from numpy import linalg as la
import math


n = 20*3

def check_symmetric(A, rtol=1e-05, atol=1e-08):
    return np.allclose(A, A.T, rtol=rtol, atol=atol)


def check(A):

    print(A)
    print("-------------")

    eigenvalues, eigenvectors = la.eig(A)

    mx = 0.0
    mi = math.inf

    if check_symmetric(A):

        print("This matrix is [symmetric].")

        mi = math.inf
        for v in eigenvalues:
            mi = min(mi, v)
            # print("eigenvalue: ", v, np.absolute(v))

        if mi > 0: print("This matrix is [positive definite].")
        else: print("This matrix is [symmetric] but **not** positive definite.")

    else:
        print("This matrix is **not** symmetric.")


    for v in eigenvalues:
        mx = max(mx, np.absolute(v))
        mi = min(mi, np.absolute(v))

    print("Spectral radius:", mx)
    print("Condition number:", mx/mi)


    # check convergence of Jacobi iteration
    S = np.zeros((n, n))
    T = np.zeros((n, n))
    for i in range(n):
        S[i, i] = A[i, i]
        for j in range(n):
            if i != j:
                T[i, j] = - A[i, j]
    B = la.inv(S)@T

    eigenvalues, eigenvectors = la.eig(B)

    mx = 0.0
    for v in eigenvalues:
        mx = max(mx, np.absolute(v))

    print("Spectral radius of B:", mx)




def read_and_check(filename):
    A = np.zeros((n, n))
    with open(filename, "r") as file:
        for i in range(n):
            row = file.readline().split()
            for j in range(n):
                A[i, j] = float(row[j])
    check(A)




read_and_check("matrix_A.txt")
