#pragma OPENCL EXTENSION cl_amd_printf : enable

__kernel void gemm_nv(__global float *A, __global float *B, __global float *C)
{
  const int g_col = get_global_id(0);
  const int g_row = get_global_id(1); 

  float result = 0.0;
  for(int i = 0; i < AW; i++)
    result += A[g_row * AW + i] * B[i * BW + g_col];
  C[g_row * BW + g_col] = result;
}

