#include "settings.h"

__kernel void hello_kernel(__global const float *a, __global const float *b,
                           __global float *result, const int N, const K)
{
  int col = get_global_id(0);
  int row = get_global_id(1);

  __local float c_val;

  c_val = 0.0f;
  for(int idx = 0; idx < K; idx++)
  {
    c_val += a[row * K + idx] * b[idx * N + col];
  }

  result[row * N + col] = c_val;
}
