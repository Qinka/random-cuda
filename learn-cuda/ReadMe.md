# Learn CUDA

This is the code that I learn CUDA and write CUDA. 

The codes are managed with CMAKE. The following list is the tested environments:
  1. Ubuntu 17.04 with CUDA 9.0 (with GTX960, 5.2)
  1. Ubuntu 16.04 with CUDA 8.0 (with GF Titan X, 5.2)*
  1. Ubuntu 16.04 with CUDA 7.5 (with GF Titan X, 5.2)*
  1. Windows 10, Visual Studio 2017(with VC++ v140) with CUDA 9.1 (with K1100M, 3.0)
  1. Ubuntu 16.04 with CUDA 8.0 (using clang-4.0, without cmake)

>  \* When the number of block is larger than 65535,
> the program will failed. However I think that it might a bug of old CUDA SDK, and the CUDA 9.x works well.



