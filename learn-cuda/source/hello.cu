#include <stdio.h>

int main() {
  int nDevices;

  cudaGetDeviceCount(&nDevices);
  printf("Hello, CUDA!\n");
  for(int i = 0; i < nDevices; ++i) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop,i);
    printf("Device number: %d\ (pci bus id: %d; pci device id: %d; pci domain id: %d)\n",i,prop.pciBusID,prop.pciDeviceID,prop.pciDomainID);
    printf("\tDevice name: %s\n",prop.name);
    printf("\tClock rate: %d kHz\n",prop.clockRate);
    printf("\tL2 cache size: %d bytes\n",prop.l2CacheSize);
    printf("\tConcurrent kernels: %d\n",prop.concurrentKernels);
    printf("\tIs multi-GPU board: %s\n",prop.isMultiGpuBoard?"yes":"no");
    printf("\tCompute capability: %d.%d\n",prop.major,prop.minor);
    printf("\tMax Grid size: %d\n", *prop.maxGridSize);
    printf("\tMax thread dim: %d\n",*prop.maxThreadsDim);
    printf("\tMax thread per block: %d\n", prop.maxThreadsPerBlock);
    printf("\tMax thread per multiprocessor: %d\n",prop.maxThreadsPerMultiProcessor);
    printf("\tMemory clock rate: %d kHz\n",prop.memoryClockRate);
    printf("\tMemory bus width: %d bits\n",prop.memoryBusWidth);
    printf("\tPeak memory bandwidth: %f GB/s\n",2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    printf("\tMemory copy pitch: %lu bytes\n",prop.memPitch);
    printf("\tManaged memory: %d\n", prop.managedMemory);
    printf("\tConst memory: %lu bytes\n",prop.totalConstMem);
    printf("\tGlobal memory: %lu bytes\n",prop.totalGlobalMem);
    printf("\n");
  }
}