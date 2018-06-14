#include "settings.h"

__kernel void gemm_local(__global const float *a, __global const float *b, __global float *result,
                         __local float Asub[LCL_SZ][LCL_SZ], Bsub[LCL_SZ][LCL_SZ])
{
  const int l_col = get_local_id(0);
  const int l_row = get_local_id(1);
  const int g_col = get_global_id(0);
  const int g_row = get_global_id(1);
  const int num_tiles = AW / LCL_SZ;

  float acc = 0.0f;
  for(int idx = 0; idx < num_tiles; idx++)
  {
    Asub[l_row][l_col] = a[g_row * AW + idx * LCL_SZ + l_col];
    Bsub[l_row][l_col] = b[(LCL_SZ * idx + l_row) * BW + g_col];

    barrier(CLK_LOCAL_MEM_FENCE);
    for(int lidx = 0; lidx < LCL_SZ; lidx++)
    {
      acc += Asub[l_row][lidx] * Bsub[lidx][l_col];
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
  result[g_row * BW + g_col] = acc;
}


