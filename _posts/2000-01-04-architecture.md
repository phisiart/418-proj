---
title: "Architecture"
bg: black
color: white
fa-icon: cogs
---

## The Architecture of TVM + OpenGL

Now let's dive in the architecture of TVM and how the OpenGL backend works. Figure 1 shows the overall flow that corresponds to the previous example.

<br/>
<center>
<img src="img/flow.png" alt="flow" style="width: 600px;"/>
</center>
<br/>
<center>
Figure 1. The Overall Flow of TVM + OpenGL
</center>
<br/>

The user starts by writing a tensor program in Python using a lambda function. The lambda function is then converted by TVM to an abstract syntax tree (AST) written in C++. This AST, shown as AST<sub>1</sub> in figure 1, is a direct mapping from the Python code, which specifies, at a high level, how to compute each element of the output tensor.

Then, the user needs to specify how the computation should be done. For example, if we are to run the program on CPU, the most naive way is to loop over all the element indices of the output tensor, and compute each element. However, we could instead **rearrange loops** in order to visit tensors block by block for better cache performance. Moreover, if we are to run the program on GPU using CUDA, we also need to decide how to **map threadIdx's** to ranges of inputs/outputs. Therefore, TVM provides the concept of a **schedule** which specifies both iteration rearrangements and threadIdx mapping. We have implemented a default OpenGL schedule, which maps each "output pixel" (similar to threadIdx) to an output element. We defer our discussion about more OpenGL-specific topics in the next section.

After a schedule is provided, TVM is able to transform AST<sub>1</sub> into AST<sub>2</sub> to explicitly express how the iterations are performed. These 2 AST's are conceptually similar to "logical plan" and "physical plan" in database terminology.

Then the user "builds" the sheduled program. The building phase is internally split into 2 stages - lowering and compiling. First, AST<sub>2</sub> is lowered into AST<sub>3</sub>, which is more like an intermediate representation. Second, based on AST<sub>3</sub>, OpenGL shader code is emitted by the **TVM OpenGL Codegen**. Again, we will defer our discussion about more OpenGL-specific topics in the next section.

Finally, the entire OpenGL program is cleanly wrapped as a single function callable from within Python. When this function gets invoked, the **TVM OpenGL Runtime** is responsible for loading the input tensors to GPU, launching the OpenGL program, and retrieving the output tensor back to CPU.
