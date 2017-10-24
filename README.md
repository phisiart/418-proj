## 15-418/618 <br/>Parallel Computer Architecture and Programming <br/>Final Project

### Zhixun Tan (zhixunt), Peng Wang (pwang1)

### Background: MXNet and TVM

[MXNet](https://github.com/apache/incubator-mxnet) is an open-source machine learning framework, similar to TensorFlow. The programmer specifies a high-level computation graph, and MXNet utilizes a data-flow runtime scheduler to execute the graph in a parallel or distributed setting, depending on the available computation resources. MXNet supports running machine learning tasks in various environments: (possibly multiple) CPUs, GPUs, or even mobile devices.

An active project within MXNet is [TVM](https://github.com/dmlc/tvm), an intermediate representation for computation graphs. After the user uses MXNet to create a machine learning program, the computation graph is transformed into a lower-level but still cross-platform representation in TVM. Then, TVM supports further transformations into platform-specific code: LLVM, CUDA, OpenCL, etc.

### Project Introduction: OpenGL Backend for TVM

Our project is to add a new backend platform for TVM: OpenGL.

#### Why OpenGL? I thought with CUDA, we are passed this age?

It is true that we can (and should) use CUDA to write neural networks for the GPU. Our goal is actually running them in the browser. Currently, WebGL is supported by all the main-stream browsers. Neither OpenCL nor CUDA is supported by them. Therefore, if we want to utilize a GPU from the browser, we still need to go back to the pre-CUDA age.

Most browsers support WebGL with GLSL (OpenGL Shading Language) v3, which does not have the "compute shader", i.e. CUDA-like programming environment. We still need to utilize the traditional graphics pipeline (with vertex shader and fragment shader).

#### Approach Overview

The basic method that we are going to use is to embed arrays inside OpenGL textures, and use fragment shaders to perform computations. In this way, OpenGL thinks we are rendering a frame of picture, but we are actually doing something like a 1D convolution.

The specific tasks we need to accomplish are:

- Create a WebGL runtime for TVM. We do have other runtimes for reference (e.g. the OpenCL runtime).

- Figure out a way to embed data into WebGL textures. Some layout issues might need to be solved. Furthermore, rumor has it that older WebGL implementations don't even support float textures.

- Perform some demo computation in the browser.

