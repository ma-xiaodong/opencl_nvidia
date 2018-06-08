#define TS	20

__kernel void hello_kernel(__global const float *a, __global const float *b,
                           __global float *result, const int N, const int K)
{
  const int l_row = get_local_id(0);
  const int l_col = get_local_id(1);
  const int g_row = get_group_id(0) * TS + l_row;
  const int g_col = get_group_id(1) * TS + l_col;
  const int num_tiles = K / TS;

  __local float Asub[TS][TS], Bsub[TS][TS];
  float acc = 0.0f;
  for(int idx = 0; idx < num_tiles; idx++)
  {
    Asub[l_row][l_col] = a[g_row * K + idx *TS + l_col];
    Bsub[l_row][l_col] = b[(TS * idx + l_row) * N + g_col];

    barrier(CLK_LOCAL_MEM_FENCE);
    for(int lidx = 0; lidx < TS; lidx++)
    {
      acc += Asub[l_row][lidx] * Bsub[lidx][l_col];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  result[g_row * N + g_col] = acc;
}


