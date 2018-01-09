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
void linearCombinKernel(float coe1, uint8_t* m1, float coe2, uint8_t* m2, int row, int col, uint8_t* m3) {
  int idxX = blockIdx.x * blockDim.x + threadIdx.x;
  int idxY = blockIdx.y * blockDim.y + threadIdx.y;
  int stdX = blockDim.x * gridDim.x;
  int stdY = blockDim.y * gridDim.y;
  for(int i = idxX; i < row; i += stdX)
    for(int j = idxY; j < col; j += stdY) {
      float tmp = coe1 * m1[i*col + j] + coe2 * m2[i*col + j];
      m3[i*col+j] = (uint8_t)(fmaxf(fminf(tmp,255),0));
    }
}


int linear_combination(float coe1, uint8_t* m1, float coe2, uint8_t* m2, int row, int col, uint8_t* m3) {
  uint8_t* dm1 = 0;
  uint8_t* dm2 = 0;
  uint8_t* dm3 = 0;
  cudaError_t rtCode;
  dim3 blocksize;
  dim3 threadsPerBlock;

  // select codes
  rtCode = cudaSetDevice(0);
  if (rtCode != cudaSuccess) {
    fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    goto Error;
  }

  // get prop
  cudaDeviceProp prop;
  rtCode != cudaGetDeviceProperties(&prop,0);
  if (rtCode != cudaSuccess) {
    fprintf(stderr, "Fail to get the device infos");
    goto Error;
  }

  // malloc
  rtCode = cudaMalloc((void**)&dm1, col * row  * sizeof(uint8_t));
  if (rtCode != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed!(m1)");
    goto Error;
  }
  rtCode = cudaMalloc((void**)&dm2, col * row  * sizeof(uint8_t));
  if (rtCode != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed!(m2)");
    goto Error;
  }
  rtCode = cudaMalloc((void**)&dm3, col * row  * sizeof(uint8_t));
  if (rtCode != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed!(m3)");
    goto Error;
  }

  // copy
  rtCode = cudaMemcpy(dm1, m1, col * row * sizeof(uint8_t), cudaMemcpyHostToDevice);
  if (rtCode != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed!(m1)");
    goto Error;
  }
  rtCode = cudaMemcpy(dm2, m2, col * row * sizeof(uint8_t), cudaMemcpyHostToDevice);
  if (rtCode != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed!(m2)");
    goto Error;
  }

  // run
  blocksize = dim3(prop.maxGridSize[0],prop.maxGridSize[1]);
  threadsPerBlock = dim3(prop.maxThreadsDim[0],prop.maxThreadsDim[1]);
  linearCombinKernel<<<blocksize,threadsPerBlock>>>(coe1,m1,coe2,m2,row,col,m3);

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
  rtCode = cudaMemcpy(m3, dm3, row * col * sizeof(int), cudaMemcpyDeviceToHost);
  if (rtCode != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed!(m3)");
    goto Error;
  }

Error:
    cudaFree(dm1);
    cudaFree(dm2);
    cudaFree(dm3);
    return rtCode;
}


#endif // _LINEAR_COMBINATION_C_
