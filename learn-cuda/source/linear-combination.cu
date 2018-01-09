/**
 * linear combination for 2-D matrix
 * Copyright 2018 (C) Johann Lee <me@qinka.pro>
 */


#ifndef _LINEAR_COMBINATION_C_
#define _LINEAR_COMBINATION_C_

#include <linear-combination.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>


__global__
void linearCombinKernel(float coe1, uint8_t* m1, float coe2, uint8_t* m2, int size, uint8_t* m3) {
  int idxX = blockIdx.x * blockDim.x + threadIdx.x;
  int stdX = blockDim.x * gridDim.x;
  for(int i = idxX; i < size; i += stdX) {
    float tmp = coe1 * m1[i] + coe2 * m2[i];
    m3[i] = (uint8_t)(fmaxf(fminf(tmp,255),0));
  }
}


int linear_combination(float coe1, uint8_t* m1, float coe2, uint8_t* m2, int _size, uint8_t* m3) {
  uint8_t* dm1 = 0;
  uint8_t* dm2 = 0;
  uint8_t* dm3 = 0;
  cudaError_t rtCode;
  int blocksize;
  int threadsPerBlock;
  size_t size = _size * sizeof(uint8_t);
  int bsX = 0;

  // select codes
  /*rtCode = cudaSetDevice(0);
  if (rtCode != cudaSuccess) {
    fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    goto Error;
    }*/

  // get prop
  cudaDeviceProp prop;
  rtCode = cudaGetDeviceProperties(&prop,0);
  if (rtCode != cudaSuccess) {
    fprintf(stderr, "Fail to get the device infos");
    goto Error;
  }

  // malloc
  rtCode = cudaMalloc((void**)&dm1, size);
  if (rtCode != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed!(m1)");
    goto Error;
  }
  rtCode = cudaMalloc((void**)&dm2,size);
  if (rtCode != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed!(m2)");
    goto Error;
  }
  rtCode = cudaMalloc((void**)&dm3, size);
  if (rtCode != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed!(m3)");
    goto Error;
  }

  // copy
  rtCode = cudaMemcpy(dm1, m1, size, cudaMemcpyHostToDevice);
  if (rtCode != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed!(m1)");
    goto Error;
  }
  rtCode = cudaMemcpy(dm2, m2, size, cudaMemcpyHostToDevice);
  if (rtCode != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed!(m2)");
    goto Error;
  }

  // run

  bsX = (size / prop.maxThreadsPerBlock) + 1;
  blocksize = min((size_t)prop.maxGridSize[0],bsX);
  threadsPerBlock = min(prop.maxThreadsPerBlock,size);
  linearCombinKernel<<<blocksize,threadsPerBlock>>>(coe1,dm1,coe2,dm2,_size,dm3);

  // check error
  rtCode = cudaGetLastError();
  if (rtCode != cudaSuccess) {
    fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(rtCode));
    goto Error;
  }

  // synchronize
  rtCode = cudaDeviceSynchronize();
  if (rtCode != cudaSuccess) {
    fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", rtCode);
    goto Error;
  }

  // copy
  rtCode = cudaMemcpy(m3, dm3, size, cudaMemcpyDeviceToHost);
  if (rtCode != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed!(m3),%d",rtCode);
    goto Error;
  }

Error:
    cudaFree(dm1);
    cudaFree(dm2);
    cudaFree(dm3);
    return rtCode;
}


#endif // _LINEAR_COMBINATION_C_
