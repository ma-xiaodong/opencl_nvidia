
__kernel void gemm_origin(__global float *A, __global float *B, __global float *C)
{
  const int g_col = get_global_id(0);
  const int g_row = get_global_id(1); 

  float result = 0.0;
  for(int i = 0; i < AW; i++)
    result += A[g_row * AW + i] * B[i * BW + g_col];
  C[g_row * BW + g_col] = result;
}

__kernel void gemm_onerow(__global float *A, __global float *B, __global float *C)
{
  const int g_row = get_global_id(0);
  const int lcl = get_local_id(0);
  int lsz = get_local_size(0);
  int j, k;

  float Asub[AW];
  __local float Bsub[AW];
  float tmp;

  for(k = 0; k < AW; k++)
    Asub[k] = A[g_row * AW + k];
  for(j = 0; j < BW; j++)
  {
    for(k = lcl; k < AW; k += lsz)
      Bsub[k] = B[k * BW + j];
    barrier(CLK_LOCAL_MEM_FENCE);
    tmp = 0.0;
    for(k = 0; k < AW; k++)
      tmp += Asub[k] * Bsub[k];
    C[g_row * AW + j] = tmp;
  }
}

__kernel void gemm_local(__global float *A, __global float *B, __global float *C)
{
  const int l_col = get_local_id(0);
  const int g_col = get_group_id(0) * BLK + l_col;
  const int l_row = get_local_id(1);
  const int g_row = get_group_id(1) * BLK + l_row;

  __local float aSub[BLK][BLK], bSub[BLK][BLK];
  float sum = 0.0f;
  const int tiles = AW / BLK;
  int idx;

  for(idx = 0; idx < tiles; idx++)
  {
    aSub[l_row][l_col] = g_row < AH? A[g_row * AW + idx * BLK + l_col]: 0.0f;
    bSub[l_row][l_col] = g_col < BW? B[(idx * BLK + l_row)* BW + g_col]: 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    for(int j = 0; j < BLK; j++)
    {
      sum += aSub[l_row][j] * bSub[j][l_col];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  idx = idx * BLK;

  aSub[l_row][l_col] = ((idx + l_col) < AW && g_row < AH)?
                       A[g_row * AW + idx + l_col]: 0.0f;
  bSub[l_row][l_col] = ((idx + l_row) < AW && g_col < BW)?
                       B[(idx + l_row) * BW + g_col]: 0.0f;
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  for(int j = 0; j < BLK; j++)
  {
    sum += aSub[l_row][j] * bSub[j][l_col];
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  if((g_row < AH) && (g_col < BW))
    C[g_row * BW + g_col] = sum;
}

