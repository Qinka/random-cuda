# Frequenlty Asked Question

* **Why does not cmake work well?**
> I hope the following code might help you for use cmake to build these codes.
> The following code define the path of `libcudart.so`, force to compile x86_64 codes(for the most machine
> do not support to compile x86), and fix the bug for might missing 
> `CMAKE_FIND_LIBRARY_SUFFIXES` and `CMAKE_FIND_LIBRARY_PREFIXES`.
> ```
> cmake -D CUDA_CUDART_LIBRARY=/usr/local/cuda/lib64/libcudart.so -D CMAKE_FIND_LIBRARY_SUFFIXES= -D CMAKE_FIND_LIBRARY_PREFIXES= -DCUDA_64_BIT_DEVICE_CODE=ON  CMAKE_FIND_LIBRARY_SUFFIXES ..
> ```
