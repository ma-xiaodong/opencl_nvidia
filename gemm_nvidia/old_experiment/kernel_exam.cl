#include "settings.h"
#pragma OPENCL EXTENSION cl_amd_printf : enable

__kernel void hello_kernel(__global const float *a, __global const float *b,
                           __global float *result, const int N, const K)
{
  int gid0 = get_global_id(0);
  int gid1 = get_global_id(1);

  int gsz0 = get_global_size(0);
  int gsz1 = get_global_size(1);

  int lsz0 = get_local_size(0); 
  int lsz1 = get_local_size(1);

  if(gid0 == 0 && gid1 == 0)
    printf("gsz0: %d, gsz1: %d, lsz0: %d, lsz1: %d.\n", gsz0, gsz1, lsz0, lsz1);
}
