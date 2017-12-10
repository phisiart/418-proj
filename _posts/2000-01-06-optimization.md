---
title: "Optimization"
bg: black
color: white
fa-icon: fighter-jet
---

## Optimization Techniques

While the fragment shader imposes a strict access pattern - we can only assign to one output element per instance of a kernel, there is still room for performance optimization.

### Using `vec4`

Since OpenGL is designed for rendering, it is natual that it provides extensive vector support. Both geometry (XYZW) and color (RGBA) require `vec4`, and it is easy to perform operations on vectors (e.g. addition, subtraction, dot product, ...).

Therefore, a natual optimization that one can think of is to utilize all the color channels in a texture. Instead of just store 1 float in each pixel (thus using only the red channel), we can store 4 floats (thus utilizing all of RGBA). This allows us to read, write, or compute 4 values at the same time.

### Using `OpenGL Intrinsics`

OpenGL also provides fast intrinsics such as `dot` and `abs`. These intrinsics can be directly computed on `vec4`'s.

TODO(pwang1): More detail on this?

### Changing Storage Layout

We can reorder the elements of tensors, such that our access pattern across nearby instances of a fragment shader is cache friendly.

TODO(pwang1): More detail on this?

#### TODO(pwang1): Other things to add?
