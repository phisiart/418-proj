## 15-418/618 <br/>Parallel Computer Architecture and Programming <br/>Final Project: WebGL Backend for MXNet/TVM

### Zhixun Tan (zhixunt), Peng Wang (pwang1)

Website: https://phisiart.github.io/418-proj/

Repository: https://github.com/phisiart/tvm

### Summary

We want to add a WebGL backend for MXNet/TVM, which enables performing tensor computations on the GPU within a browser.

### Background: MXNet and TVM

[MXNet](https://github.com/apache/incubator-mxnet) is an open-source deep learning framework, similar to [TensorFlow](https://github.com/tensorflow/tensorflow), [Caffe](https://github.com/caffe2/caffe2), [CNTK](https://github.com/Microsoft/CNTK), etc. The programmer specifies a high-level **computation graph**, and MXNet utilizes a data-flow runtime scheduler to execute the graph in a parallel / distributed setting, depending on the available computation resources. MXNet supports running deep learning algorithms in various environments: CPUs, GPUs, or even mobile devices.

An active project within MXNet is [TVM](https://github.com/dmlc/tvm), an intermediate representation for computation graphs. After the user uses MXNet (or other frameworks that TVM intends to support) to create a machine learning program, the computation graph is transformed into a lower-level but still cross-platform representation in TVM. Then, TVM supports further transformations into platform-specific code: CUDA, OpenCL, etc. In other words, TVM is considered the LLVM for deep learning.

### Project Introduction: OpenGL Backend for TVM

Our project is to add a new backend platform for TVM: OpenGL. More specifically, we want TVM to be able to generate GLSL (OpenGL shading language) kernels to perform tensor computations on the GPU.

#### Why OpenGL when we have CUDA?

A natural question is why we want to use OpenGL instead of CUDA (or OpenCL) to perform computation on the GPU. It is true that we can (and should) use CUDA to write neural networks for the GPU. Our goal is actually running them in a browser. Currently, WebGL is supported by all the main-stream browsers, but neither OpenCL nor CUDA is supported by them. Therefore, if we want to utilize a GPU from the browser, we still need to go back to the dark pre-CUDA age.

Most browsers support WebGL with GLSL (OpenGL Shading Language) v3, which does not have the "compute shader", i.e. CUDA-like computation kernel. We still need to utilize the traditional graphics pipeline (with a vertex shader and a fragment shader).

#### Project Outline

The main idea is to use OpenGL fragment shaders. We embed input arrays inside OpenGL textures, and perform computation within the shader. In this way, OpenGL thinks we are rendering a frame of picture, but we are actually doing tensor computations like matrix multiplication or 1D convolution.

In order to achieve our goal, we need to accomplish the following:

- Codegen for OpenGL. We need to generate OpenGL fragment shaders from the TVM AST (abstract syntax tree). Currently TVM has codegen for C (yes, plain C code), CUDA, and OpenCL. The latter 2 reuse much of the code from the first, and cherry pick the different parts. We will do the same for OpenGL.

- Scheduler for OpenGL. This is a TVM-specific term. The codegen needs to understand things like "blockIdx" and "threadIdx" in order to generate kernels. Unlike CUDA kernels which are per-thread, fragment shaders are per-pixel. Therefore, we need to add specific scheduling code to deal with this.

- Runtime for OpenGL. This is the piece of code that loads input arrays into textures, launches the shader, and retrieves the output. TVM has an OpenCL runtime, and we need to write a similar one for OpenGL.

- (Optional) JavaScript runtime for WebGL. In order to perform computations in the browser, we need to launch shaders using WebGL, in JavaScript, from a browser. This means we need to create another JavaScript runtime.

### Technical Challenges

- OpenGL assumes we are rendering an image. This means the data layouts are specially designed for rendering tasks. For example, the textures are 1D/2D/3D arrays of RGBA structures. We need to perform data layout transformations to utilize memory.

- The GLSL fragment shader poses limits on data access patterns. More specifically, inside the kernel, we can only **assign to one output pixel** (i.e. the fragment shader is a per-pixel kernel). Luckily, we do have the ability to **read from arbitrary input texture locations**.

- WebGL1 without extensions only supports `int8` textures. This means we can only use `int8` input arrays. WebGL2 does support `float` textures by default, and are supported by Chrome and Firefox. If we are to support WebGL1, we need extra encoding/decoding operations.

### Resources and Reference

- We will work on the codebase of [TVM (https://github.com/dmlc/tvm)](https://github.com/dmlc/tvm). TVM already has codegen and runtime for OpenCL, which should be helpful reference for our OpenGL codegen and runtime.

- Some handcrafted WebGL kernels for specific algorithms are [https://github.com/waylonflinn/weblas](https://github.com/waylonflinn/weblas) and [https://github.com/PAIR-code/deeplearnjs/tree/master/src/math/webgl](https://github.com/PAIR-code/deeplearnjs/tree/master/src/math/webgl). We want TVM to automatically generate OpenGL shaders that have similar or better performance.

- [This document (http://www.seas.upenn.edu/~cis565/fbo.htm)](http://www.seas.upenn.edu/~cis565/fbo.htm) provides an introduction to using OpenGL for GPGPU computations.

### Goals and Deliverables

- [75%] A standalone (without TVM) program that uses OpenGL to do some computation. This program should have one or more array inputs, and an array output. This helps us get used to the overall OpenGL workflow (how to embed inputs into textures, and how to use fragment shaders).

- [100%] Codegen for OpenGL that can generate fragment shaders (aka kernels).

- [100%] C++ Runtime backend that deals with memory allocation/copy, etc., which sets up the environment to execute the shaders.

- [100%] Explore the TVM scheduler for optimization (we need to further our understanding of TVM before determining what/how to optimize).

- [125%] WebGL backend in JavaScript that runs in the browser.

### Platforms of Choice

The core of MXNet/TVM is written in modern C++. It can be invoked from a Python API. We expect that the core part of our code should be in C++.

We will write a code generator that generates GLSL kernels.

In order to make our GLSL kernels run in the browser, we also need to write some JavaScript (or TypeScript if possible?).

Our code should run on any computer that has a GPU and supports OpenGL.

### (Tentative) Schedule

- Week 1 (Until 11/05):

  Create a standalone OpenGL program.

  Study the structure of TVM.

- Week 2 (Until 11/12)

  Study the structure of OpenCL codegen and runtime of TVM.

  Prototype OpenGL codegen and runtime for TVM. The components should be able to generate and run a simple example OpenGL program. The OpenGL program doesn't run in the browser yet.

- Week 3 (Until 11/19)

  Extend the OpenGL codegen to support more operations.

- Week 4 (Until 11/26)

  Explore TVM scheduler for optimization.

- Week 5 (Until 12/03)

  Port the runtime onto the browser.

- Week 6 (Until 12/10)

  Finish up anything left.

## Checkpoint - OpenGL Backend for MXNet/TVM

### Current Progress

Due to interviews and midterms, we are behind original schedule by ~ 1.5 week.

Frankly speaking, with the limited amount of time, we are happy that we are able to achieve the following.

- A simple standalone program that:
  - Has no binary library dependencies (no apt-get/brew needed);
  - Is cross-platform (Linux and Mac);
  - Initializes OpenGL 3.3 (the version similar to WebGL 2);
  - Can load float arrays into OpenGL textures;
  - Can retrieve float arrays back from OpenGL textures;
  - Can set up a framebuffer and bind a texture to render on;
  - Can build an OpenGL program given a fragment shader;
  - Can launch an OpenGL program to render.

  The program is here:

  https://github.com/phisiart/Glitter/blob/master/Glitter/Sources/main.cpp

  In short, this program does exactly everything that the TVM OpenGL runtime would need to do.
  
  We have run the following simple program as a test:
  ```
  C[x, y] = A[x, y] + B[x, y]
  ```

  The above program yields the correct result.

  We set all the matrices to have size 2500x2500. Then running OpenGL yields a 9x speedup vs CPU. Notice that this example is very simple, and is memory bound. We expect more speedup for more complicated kernels.

- TVM:
  https://github.com/phisiart/tvm

  - The skeleton codegen + runtime for the OpenGL backend;

  - Runs a test program that doesn't give the correct result (of course because its just a skeleton, but the key is that the whole flow can be completed with no runtime error).

### Future Schedule

Week 11/20 - 11/26
- Migrate the standalone program into the TVM OpenGL runtime (pwang1).
  
  After doing this, TVM OpenGL should pass this test case:
  
  https://github.com/phisiart/tvm/blob/opengl/tests/python/unittest/test_runtime_ndarray.py

- Implement OpenGL codegen to generate a simplest program (zhixunt).

  After doing this, TVM OpenGL should codegen and run this program:

  ```
  C[x, y] = A[x, y] + B[x, y]
  ```

Week 11/27 - 12/03
- Explore 1 optimization method for TVM OpenGL (pwang1).

- Extend OpenGL codegen to support more AST nodes (zhixunt).

Week 12/04 - 12/10
- Explore 1 more optimization method for TVM OpenGL (pwang1).

- Sum up the project (zhixunt).

- (Optional) Sketch WebGL on browser.

### Goals

- TVM OpenGL codegen

- TVM OpenGL runtime

- 1 ~ 2 Optimization for TVM OpenGL

- (Optional) WebGL on browser

### Poster Session

We would like to provide a demo, which shows the typical flow of using TVM through a very simple example:

- Write a tensor program in Python;
- Create a schedule for OpenGL;
- Codegen fragment shader;
- Launch the OpenGL shader program;
- Read the result in Python.

### Concerns

The biggest concern is that the fragment shader doesn't really leave too much room for optimization. At least not in the sense of splitting arrays into "block"s.

Some areas I can think of to explore are:

- We are always using 2D textures, even when our arrays are 1D.

  Do 2500x2500 and 5000x1250 have different performance?

- We are passing in textures as `uniform sampler2D`'s.

  When we calculate `C[x, y] = A[x, y] + B[x, y]`, can we utilize the OpenGL texture mapping directly?
