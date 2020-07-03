## Introduction
An implementation of 3D linear FEM based on Neohooken elasticity model.


Three different numerical method are implemented respectively, include:
- Explicit Time Integration
- Implicit Time Integration
    - Jacobi Method
    - Conjugate Gradient Method

Environment: `Taichi 0.6.15`
<!-- Comparison between three method is presented as below: -->

## Simulation Results


<!-- <img src="rendered/out-ja.gif" width="20%" > -->

![](rendered/out-exp.gif)
![](rendered/out-ja.gif)
![](rendered/out-cg.gif)

Rendered by Blender.

## Analysis

<!-- ### Same Scene -->
<!-- ![](analysis/same/out-0-exp.gif) -->
<!-- ![](analysis/same/out-0-ja.gif) -->
<!-- ![](analysis/same/out-0-cg.gif) -->

### Large Time Step
 ![](analysis/time_step/out-1-exp.gif)
 ![](analysis/time_step/out-1-ja.gif)
 ![](analysis/time_step/out-1-cg.gif)

### Large Young's Modulus
 ![](analysis/Young_modulus/out-2-exp.gif)
 ![](analysis/Young_modulus/out-2-ja.gif)
 ![](analysis/Young_modulus/out-2-cg.gif)

### Problems of Conjugate Gradient Method

<!-- 1. Rotation -->
 ![](analysis/problems/out-cg-rotation.gif)
 ![](analysis/problems/out-cg-explosion.gif)


<!-- | Tables   |      Are      |  Cool | -->
<!-- |----------|:-------------:|------:| -->
<!-- | col 1 is |  left-aligned | $1600 | -->
<!-- | col 2 is |    centered   |   $12 | -->
<!-- | col 3 is | right-aligned |    $1 | -->
