---
title: "Experiments"
bg: gray
color: white
fa-icon: list
---

## Experiments

We implemented all the optimizations mentioned in the last section, and tested their running time with as standalone programs. Since the variables and the compiled TVM functions will be similarly executed, we expect the performance would be similar if the programs were integrated in TVM + OpenGL.

The program we study calculates the product of two square matrices of side length `N`. For each `N`, the actual calculation is repeated a number of times and their average running time is recorded.

### Configurations

The programs were run on two machines with the following configurations:

- iMac: Intel Core i7-4790K (4 cores, 8 threads, 4.0 GHz, turbo up to 4.4 GHz). 16 GB RAM (2 * DDR3, 1600 MHz). AMD Radeon R9 M295X (850 MHz, 4 GB GDDR5).

- MacBook Pro (MBP): Intel Core i5-5287U (2 cores, 4 threads, 2.9 GHz, turbo up to 3.3 GHz). 8 GB RAM (2 * LPDDR3, 1867 MHz). Intel Iris Graphics 6100 (300 MHz, up to 1.1 GHz, 4 GB GDDR).

The following names are used to identify the corresponding implementation:

- 2D, CPU: Matrices are represented as 2D matrices in row-major order, and use a simple loop to calculate for each cell.

- 2D, CPU, O3: "2D, CPU" compiled with `gcc`'s `-O3` flag.

- 1D, CPU, O3: Matrices are stored as long, 1D arrays. The position of cells is manually converted to indices. `-O3` flag is applied.

- OpenCL CPU: The 2D logic translated to OpenCL, and computed on CPU.

- 1D, GPU: Matrices stored as 1D arrays. Computed with OpenGL.

- 2D, GPU: Matrices stored as 2D arrays. Computed with OpenGL.

- Trans: 2D arrays with the second matrix stored in column-major order. Computed with OpenGL.

- OpenCL GPU: The 2D logic translated to OpenCL, and computed on GPU.

- Vec4: As described in "Using `vec4`".

- Dot: As described in "Using OpenGL Intrinsics".
