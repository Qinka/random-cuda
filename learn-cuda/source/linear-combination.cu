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
  size_t size = col * row  * sizeof(uint8_t);
  int bsX = 0;
  int bsY = 0;

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
  bsX = (row / prop.maxThreadsDim[0]) + 1;
  bsY = (col / prop.maxThreadsDim[1]) + 1;
  blocksize = dim3(min(prop.maxGridSize[0],bsX),min(prop.maxGridSize[1],bsY));
  threadsPerBlock = dim3(min(prop.maxThreadsDim[0],row),min(prop.maxThreadsDim[1],col));
  linearCombinKernel<<<blocksize,threadsPerBlock>>>(coe1,dm1,coe2,dm2,row,col,dm3);

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
