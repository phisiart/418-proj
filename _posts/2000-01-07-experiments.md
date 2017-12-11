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

- OpenCL CPU: The loop-addition logic translated to OpenCL, and computed on CPU.

- 1D, GPU: Matrices stored as 1D arrays. Computed with OpenGL.

- 2D, GPU: Matrices stored as 2D arrays. Computed with OpenGL.

- Trans: 2D arrays with the second matrix stored in column-major order. Computed with OpenGL.

- OpenCL GPU: The loop-addition logic translated to OpenCL, and computed on GPU.

- Vec4: As described in "Using `vec4`".

- Dot: As described in "Using OpenGL Intrinsics".

### Results

These two plots demonstrate the running time of these programs on the two machines, with different matrix sizes.

<br/>
<center>
<img src="img/imac.png" alt="imac" style="width: 600px;"/>
</center>
<br/>
<center>
Figure 3. Running time on iMac
</center>
<br/>

<br/>
<center>
<img src="img/mbp.png" alt="imac" style="width: 600px;"/>
</center>
<br/>
<center>
Figure 4. Running time on MBP
</center>
<br/>

Note: Due to OpenGL's limits on the size of each dimension of textures, the 1D solutions are only tested with `N<=128`.

### Discussion

The two platforms used in this experiment (iMac and MBP) represent two major categories: discrete (AMD Radeon) and integrated (Intel Iris) video cards. We expect NVIDIA video cards to have similar performance characteristics with AMD ones, which are found in iMacs. We plan to do the comparison on NVIDIA cards and with CUDA in the future.

Generally, as the size of matrices grow, the speed of different implementations on GPU can be ranked as: 1D < 2D = Trans < OpenCL < Vec4 = Dot (for iMac); and 1D < Trans < 2D = OpenCL < Vec4 = Dot (for MBP). In other words, the straightforward OpenGL-based solution is slower than OpenCL, but with our proposed optimizations, the OpenGL kernel can be several (2 to 9) times faster than OpenCL.

The first thing to observe is that 2D assignment of work is better than 1D, as can be seen from "1D GPU" versus "2D GPU". We are not aware of how exactly the OpenGL implementation assign work to blocks. But as reasoned in the last section, by informing OpenGL the 2D positional information of cells, it enables the scheduler to assign work more cache-friendly. This effect is more prominent on Intel Iris, and is possibly related to the difference in memory hierarchies and schedulers. 

Then, in both graphs we notice that the performance is roughly the same whether matrices are stored in row-major or column-major order. This is expected. Since both GPUs have the notion of "warps", and adjacent cells will be calculated on different cores simultaneously, the cache line is still fully utilized when reading the column vector from a row-major matrix.

Thirdly, the OpenCL programs are faster than the simple OpenGL kernel when `N` is relatively large, but slower for small `N`'s because of possible set-up overhead. The former can be 3 times faster for AMD Radeon, but in the case of Intel Iris the difference is relatively small. We suspect this is related to how the vendors implement OpenCL. But in general, OpenGL (with our choice of version) is intended for images, and involves more steps in the rendering pipeline; while OpenCL is designed for general-purpose computation.

Finally, we are glad to see that with our optimizations (namely `vec4` and `dot`), the OpenGL programs outperform OpenCL on both platforms. This should largely be attributed to the SIMD inside each execution unit (i.e. each pixel/cell). The OpenGL `vec4` structure allows us to fetch and compute 4 elements with a single intrinsic. It seems that the compilers for both OpenGL and OpenCL compilers, by default, do not use SSE (or similar technologies) for the loop that calculates the dot product. Therefore, our hand-written optimizations will help.

In conclusion, the experiments show our optimizations for OpenGL are very efficient in performance. These techniques can be integrated into the TVM codegen and runtime immediately, without the need to change anything in the user's programs. Since the performance characteristics of OpenGL is platform-specific, we will carry out more experiments on other targets.
