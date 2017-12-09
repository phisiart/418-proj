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
