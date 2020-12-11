# Hand
Hand is a minimal neural network library. Its primary use at this point is for learning the core workings of neural networks and experimenting with new techniques. Development on this library focuses on simple design to ease future maintainability, while allowing for easy extension and experimentation.

To operate at the speed of other libraries, the library is based on pytorch Tensors. Tensors interfaces with a low level BLAS (Basic Linear Algebra Subprograms) implementation, offering a massive speedup over using pure python for such operations. Tensors also provide the ability to interface with a GPU through CUDA. This allows us to run linear algebra operations at optimized C speed using BLAS, and perform operations on the GPU without having to worry about the low level implementation of such operations.

# Setup

`git clone https://github.com/DKdekes/hand-nn.git`

`cd hand-nn`

`pip install -r requirements.txt`

To test your setup by running iris.py:


To test, clone it and run examples/bit_main.py. The library will initialize a network and train it to predict the base 10 value of input binary numbers.

# Getting Started

[torch.Tensor documentation](https://pytorch.org/docs/stable/tensors.html)
- It's good to have a good understanding of the Tensor if working in this library. It very functionally similar to numpy's ndarray and is the foundation for this library.
