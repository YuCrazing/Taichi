## Introduction
An implementation of 3D linear FEM based on Neohookean elasticity model (No damping force).

Three different numerical method are implemented respectively, include:
- Explicit time integration
- Implicit time integration, solved by
    - Jacobi Method
    - Conjugate Gradient Method

Environment: `Taichi 0.6.15`

## Simulation Results


![](rendered/out-exp.gif)
![](rendered/out-ja.gif)
![](rendered/out-cg.gif)

Using `Explicit`, `Jacobi`  and `CG` method respectively (Rendered by Blender).

## Analysis

For `Explicit` method, there is no constraints between nodes when updating, so the system explodes if time step or Young's modulus is large.

For `Jacobi` method, the convergence is determined by the spectral radius `\rho` of the matrix `-D^{-1}(L+U)` , Jacobi method converges if and only if `\rho` is less than `1`.

For `CG` method, the system always converges since the coefficient matrix `A` is positive definite for (solid) FEM problems.

### Same Scene
 ![](analysis/same/0-exp.gif)
 ![](analysis/same/0-ja.gif)
 ![](analysis/same/0-cg.gif)

`Explicit` and `Jacobi` method should update many times using a small time step at each frame.

`\rho = 0.0036`

### Large Time Step
 ![](analysis/time_step/1-exp.gif)
 ![](analysis/time_step/1-ja.gif)
 ![](analysis/time_step/1-cg.gif)

`\rho = 1.09`


### Large Young's Modulus
 ![](analysis/Young_modulus/2-exp.gif)
 ![](analysis/Young_modulus/2-ja.gif)
 ![](analysis/Young_modulus/2-cg.gif)

`\rho = 1.13`



### Problems of Conjugate Gradient Method

 ![](analysis/problems/3-cg.gif)

1. Object rotates slowly when time step is large.

This problem is caused by the lack of damping force, add damping force to object can avoid this problem.

 ![](analysis/problems/4-cg.gif)

2. CG doesn't converge when both time step and Young's modulus are large.

I still don't know why CG doesn't converge in this condition, I guess this is caused by numerical accuracy.



<!-- ## Reference -->


<!-- | Tables   |      Are      |  Cool | -->
<!-- |----------|:-------------:|------:| -->
<!-- | col 1 is |  left-aligned | $1600 | -->
<!-- | col 2 is |    centered   |   $12 | -->
<!-- | col 3 is | right-aligned |    $1 | -->
