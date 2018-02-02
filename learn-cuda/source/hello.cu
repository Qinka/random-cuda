#include <iostream>

using std::cout;
using std::endl;



int main() {
  int nDevices;

  cudaGetDeviceCount(&nDevices);
  cout << "Hello, CUDA!" << endl;
  for(int i = 0; i < nDevices; ++i) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop,i);
    cout << "Device number: " << i
         << " (pci bus id: " << prop.pciBusID
         << "; pci device id: " << prop.pciDeviceID
         << "; pci domain id: " << prop.pciDomainID
         << ")" << endl;
    cout << "\tDevice name: " << prop.name << endl;
    cout << "\tClock rate: " << prop.clockRate << " kHz" << endl;
    cout << "\tL2 cache size: " << prop.l2CacheSize << "bytes" << endl;
    cout << "\tConcurrent kernels: " << prop.concurrentKernels << endl;
    cout << "\tIs multi-GPU board: " << (prop.isMultiGpuBoard ? "yes" : "no") << endl;
    cout << "\tCompute capability: " << prop.major << "." << prop.minor << endl;
    cout << "\tMax Grid size(x,y,z): "
         << prop.maxGridSize[0] << ", "
         << prop.maxGridSize[1] << ", "
         << prop.maxGridSize[2] << endl;
    cout << "\tMax thread dim: "
         << prop.maxThreadsDim[0] << ", "
         << prop.maxThreadsDim[1] << ", "
         << prop.maxThreadsDim[2] << endl;
    cout << "\tMax thread per block: " << prop.maxThreadsPerBlock << endl;
    cout << "\tMax thread per multiprocessor: " << prop.maxThreadsPerMultiProcessor << endl;
    cout << "\tMemory clock rate: " << prop.memoryClockRate << " kHz" << endl;
    cout << "\tMemory bus width: " << prop.memoryBusWidth << " bits" << endl;
    cout << "\tPeak memory bandwidth: "
         << 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6
         << " GB/s" << endl;
    cout << "\tMemory copy pitch: " << prop.memPitch << " bytes" << endl;
    cout << "\tManaged memory: " << prop.managedMemory << endl;
    cout << "\tConst memory: " << prop.totalConstMem << " bytes" << endl;
    cout << "\tGlobal memory: " << prop.totalGlobalMem << " bytes" << endl;
    cout << "\tShared memory per block: " << prop.sharedMemPerBlock << endl;
#if __CUDA_ARCH__ >= 900
    cout << "\tPer device maximum shared memory per block usable by special opt in: "
         << prop.sharedMemPerBlockOptin << endl;
#endif
    cout << "\tShared memory available per multiprocessor in bytes: "
         << prop.sharedMemPerMultiprocessor << endl;
    cout << "\tCores: " << prop.multiProcessorCount << endl;
    cout << "\tNumber of asynchronous engines: " << prop.asyncEngineCount << endl;
    cout << "\tDevice can map host memory with cudaHostAlloc/cudaHostGetDevicePointer:"
         << (prop.canMapHostMemory ? "yes" : "no") << endl;
#if __CUDA_ARCH__ >= 900
    cout << "\tDevice can access host registered memory at the same virtual address as the CPU"
         << prop.canUseHostPointerForRegisteredMem << endl;
#endif
    cout << "\tComputing mode: " << prop.computeMode << endl;

#if __CUDA_ARCH__ >= 900
    cout << "\tDevice supports Compute Preemption: " << prop.computePreemptionSupported << endl;
#endif
    cout << "\tDevice can possibly execute multiple kernels concurrently: "
         << prop.concurrentKernels << endl;
    cout << "\tDevice can coherently access managed memory concurrently with the CPU: "
         << prop.concurrentManagedAccess << endl;

#if __CUDA_ARCH__ >= 900
    cout << "\tDevice supports launching cooperative kernels via cudaLaunchCooperativeKernel: "
         << prop.cooperativeLaunch << endl;
    cout << "\tDevice can participate in cooperative kernels launched via cudaLaunchCooperativeKernelMultiDevice "
         << prop.cooperativeMultiDeviceLaunch << endl;
#endif
    cout << "\tDevice can concurrently copy memory and execute a kernel. Deprecated. Use instead asyncEngineCount: "
         << prop.deviceOverlap << endl;
    cout << "\tDevice has ECC support enabled: " << (prop.ECCEnabled ? "yes" : "no") << endl;
    cout << "\tDevice supports caching globals in L1:"
         << (prop.globalL1CacheSupported ? "yes" : "no") << endl;
    cout << "\tLink between the device and the host supports native atomic operations"
         << (prop.hostNativeAtomicSupported ? "yes" : "no") << endl;
    cout << "\tDevice is integrated as opposed to discrete:"
         << (prop.integrated ? "yes" : "no") << endl;
    cout << "\tSpecified whether there is a run time limit on kernels"
         << prop.kernelExecTimeoutEnabled << endl;
    cout << "\tDevice supports caching locals in L1: "
         << (prop.localL1CacheSupported ? "yes" : "no") << endl;
    cout << "\tUnique identifier for a group of devices on the same multi-GPU board: "
         << prop.multiGpuBoardGroupID << endl;
    cout << "\tDevice supports coherently accessing pageable memory without calling cudaHostRegister on it: " << (prop.pageableMemoryAccess ? "yes" : "no") << endl;
    cout << "\t32-bit registers available per block: " << prop.regsPerBlock << endl;
    cout << "\t32-bit registers available per multiprocessor"
         << prop.regsPerMultiprocessor << endl;
    cout << "\tRatio of single precision performance (in floating-point operations per second) to double precision performance"
         << prop.singleToDoublePrecisionPerfRatio << endl;
    cout << "\tDevice supports stream priorities"
         << (prop.streamPrioritiesSupported ? "yes" : "no") << endl;
    cout << "\tWhether device is a Tesla device using TCC driver:"
         << (prop.tccDriver ? "yes" : "no") << endl;
    cout << "\tDevice shares a unified address space with the host: "
         << prop.unifiedAddressing << endl;
    cout << "\tWarp size in threads " << prop.warpSize << endl;
    cout << endl;
  }
}
