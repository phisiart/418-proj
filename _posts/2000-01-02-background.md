---
title: "Introduction"
bg: black
color: white
fa-icon: book
---

## Background: MXNet and TVM

[MXNet](https://github.com/apache/incubator-mxnet) is an open-source deep learning framework, similar to [TensorFlow](https://github.com/tensorflow/tensorflow), [Caffe](https://github.com/caffe2/caffe2), [CNTK](https://github.com/Microsoft/CNTK), etc. The programmer specifies a high-level **computation graph**, and MXNet utilizes a data-flow runtime scheduler to execute the graph in a parallel / distributed setting, depending on the available computation resources. MXNet supports running deep learning algorithms in various environments: CPUs, GPUs, or even mobile devices.

An active project within MXNet is [TVM](https://github.com/dmlc/tvm), an **intermediate representation** for tensor computation. After the user uses MXNet (or other frameworks that TVM intends to support) to create a machine learning program, the computation graph is transformed into a lower-level but still cross-platform representation in TVM. Then, TVM supports further transformations into platform-specific code: CUDA, OpenCL, etc. In other words, TVM is considered the LLVM for deep learning.

## Our Project: OpenGL Backend for TVM

In our project, we added **a new backend platform for TVM: OpenGL**. More specifically, we made TVM able to generate OpenGL shading language (GLSL) kernels to perform tensor computation on the GPU.

### Why OpenGL when we have CUDA?

A natural question is why we want to use OpenGL instead of CUDA (or OpenCL) to perform computation on the GPU.

While it is true that we can (and should) use CUDA to write GPGPU programs, CUDA is not present in many platforms. On the other hand, OpenGL is a widely supported framework: it's supported on desktop computers, mobile devices and even browsers.

Table 1 lists the support of different frameworks on common platforms.

<br/>
<center>
<img src="img/frameworks.png" alt="frameworks" style="width: 600px;"/>
</center>
<br/>
<center>
Table 1. Frameworks on Various Platforms
</center>
<br/>

We must be very careful when using OpenGL because different platforms have different variants of it. A desktop computer has the normal full OpenGL; a mobile device has OpenGL ES; and a browser has WebGL.

In order to maximize compatibility, we looked into the features of WebGL2, which is supported by [all the main-stream browsers](https://caniuse.com/#feat=webgl2). One key discovery is that it does not yet support the new **compute shader**, e.g. CUDA-like kernel. Therefore, we must stick to the traditional rendering pipeline, and manually figure out a way to map general tensor computation into rendering tasks.

