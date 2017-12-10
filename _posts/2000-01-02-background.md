---
title: "Introduction"
bg: black
color: white
fa-icon: book
---

### Background: MXNet and TVM

[MXNet](https://github.com/apache/incubator-mxnet) is an open-source deep learning framework, similar to [TensorFlow](https://github.com/tensorflow/tensorflow), [Caffe](https://github.com/caffe2/caffe2), [CNTK](https://github.com/Microsoft/CNTK), etc. The programmer specifies a high-level **computation graph**, and MXNet utilizes a data-flow runtime scheduler to execute the graph in a parallel / distributed setting, depending on the available computation resources. MXNet supports running deep learning algorithms in various environments: CPUs, GPUs, or even mobile devices.

An active project within MXNet is [TVM](https://github.com/dmlc/tvm), an **intermediate representation** for tensor computation. After the user uses MXNet (or other frameworks that TVM intends to support) to create a machine learning program, the computation graph is transformed into a lower-level but still cross-platform representation in TVM. Then, TVM supports further transformations into platform-specific code: CUDA, OpenCL, etc. In other words, TVM is considered the LLVM for deep learning.

### Our Project: OpenGL Backend for TVM

In our project, we added **a new backend platform for TVM: OpenGL**. More specifically, we made TVM able to generate OpenGL shading language (GLSL) kernels to perform tensor computation on the GPU.

#### Why OpenGL when we have CUDA?

A natural question is why we want to use OpenGL instead of CUDA (or OpenCL) to perform computation on the GPU.

While it is true that we can (and should) use CUDA to write GPGPU programs, our end goal is actually running them in a browser. Currently, **WebGL** is supported by [all the main-stream browsers](https://caniuse.com/#feat=webgl2), but neither OpenCL nor CUDA is supported by them. Therefore, if we want to utilize a GPU from the browser, we still need to go back to the dark pre-CUDA age.

Most browsers support WebGL with GLSL (OpenGL Shading Language) v3, which does not have the "compute shader", i.e. CUDA-like computation kernel. We still need to utilize the traditional graphics pipeline (with a vertex shader and a fragment shader).
