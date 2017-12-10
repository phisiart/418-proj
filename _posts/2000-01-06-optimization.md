---
title: "Optimization"
bg: black
color: white
fa-icon: fighter-jet
---

## Optimization Techniques

While the fragment shader imposes a strict access pattern - we can only assign to one output element per instance of a kernel, there is still room for performance optimization.

In particular, we are interested in optimizing a fundamental computation task needed in machine learning - matrix multiplication.

### Using `vec4`

Since OpenGL is designed for rendering, it is natual that it provides extensive vector support. Both geometry (XYZW) and color (RGBA) require `vec4`, and it is easy to perform operations on vectors (e.g. addition, subtraction, dot product, ...).

Therefore, a natual optimization that one can think of is to utilize all the color channels in a texture. Instead of just store 1 float in each pixel (thus using only the red channel), we can store 4 floats (thus utilizing all of RGBA). This allows us to read, write, or compute 4 values at the same time.

### Using OpenGL Intrinsics

OpenGL also provides fast intrinsics such as `dot` and `abs`. These intrinsics can be directly computed on `vec4`'s.

In our case, the computation task for a pixel represented by a `vec4` is the dot product of a row and 4 columns. Therefore, we use the `dot` intrinsic for every `4*4` block.

### Changing Storage Layout

We can reorder the elements of tensors, such that our access pattern across nearby instances of a fragment shader is cache friendly.

When we compute the product of two matrices, say `C=A*B`, normally both `A` and `B` are both stored in row-major order. The kernel for each element in `C` will access a row from `A`, which has good spatial locality; but it will read from a column from `B`, whose address is not contiguous.

If the size of a cache line is larger than a float, this will be bad for the cache local to this thread. However, if the GPU execute kernels in a SIMD fashion like CUDA's warp, then the kernels in the same warp will access adjacent columns in the same time, and so they can still share the data in their block-local cache.

Another choice is to give users the option to specify that matrix `B` should be stored in column-major order, so it will be faster when appearing on the RHS of a multiplication. Then the kernel only needs to go to its thread-local cache.

### Using 2D texture versus 1D texture

In our example in the previous section of matrix addition, the input and output matrices are represented as linear vectors. When we adopt the kernel to multiplication, we will need to do the manual conversion between 1D and 2D indices. The potential drawback lies in cache locality rather than the little extra computation. 

Say the warp size is `16`. Then in the case of 1D texture, for each warp's worth of task, `1` row and `16` column vectors will be fetched. On the other hand, had the cells been arranged into `4*4`blocks, we would only need to fetch `4` rows and `4` columns. Note the rows will be used again for the next warp immediately, so this could almost effectively save us `sqrt(warp_size)` times of read.

However, this does not apply to all varieties of hardware, and we cannot control whether/how the OpenGL implementation assigns blocks.
