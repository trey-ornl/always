#pragma once

#include <cuda_runtime.h>

#define hipError_t cudaError_t
#define hipFree cudaFree
#define hipGetDevice cudaGetDevice
#define hipGetDeviceCount cudaGetDeviceCount
#define hipGetErrorName cudaGetErrorName
#define hipGetErrorString cudaGetErrorString
#define hipHostFree cudaFreeHost
#define hipHostMalloc cudaHostAlloc
#define hipHostMallocDefault cudaHostAllocDefault
#define hipMalloc cudaMalloc
#define hipMemcpy cudaMemcpy
#define hipMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define hipMemcpyHostToDevice cudaMemcpyHostToDevice
#define hipMemGetInfo cudaMemGetInfo
#define hipMemset cudaMemset
#define hipSetDevice cudaSetDevice
#define hipSuccess cudaSuccess
