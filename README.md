# Hand NN 
A simple neural network library built using only pytorch Tensors. It can train and get good accuracy on the iris dataset. To test, clone it and run examples/iris.py. MNIST coming soon.

The library uses pytorch Tensors to utilize the low level BLAS (Basic Linear Algebra Subprograms) capability that Tensors provide. Tensors also provide the ability to perform operations on the GPU through CUDA. This allows us to run linear algebra operations at C speed using BLAS, and perform operations on the GPU without having to worry about the low level implementation of such operations. 


https://github.com/pytorch/pytorch:
Description, design, value
Installation
Getting Started
