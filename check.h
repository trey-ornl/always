#pragma once
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <hsa/hsa_ext_amd.h>

static void checkAMD(const hipError_t err, const char *const file, const int line)
{
  if (err == hipSuccess) return;
  fprintf(stderr,"HIP ERROR AT LINE %d OF FILE '%s': %s %s\n",line,file,hipGetErrorName(err),hipGetErrorString(err));
  fflush(stderr);
  exit(err);
}

static void checkAMD(const hsa_status_t err, const char *const file, const int line)
{
  if (err == HSA_STATUS_SUCCESS) return;
  const char *string;
  hsa_status_string(err,&string);
  fprintf(stderr,"HSA ERROR AT LINE %d OF FILE '%s': %d %s\n",line,file,err,string);
  fflush(stderr);
  exit(err);
}

#define CHECK(X) checkAMD(X,__FILE__,__LINE__)

