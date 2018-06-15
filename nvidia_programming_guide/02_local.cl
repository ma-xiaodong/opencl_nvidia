__kernel void local_opt(const __global float *A, const __global float *B, 
                        __global float *C)
{
    const int l_col = get_local_id(0);
    const int l_row = get_local_id(1);
    const int g_col = get_global_id(0);
    const int g_row = get_global_id(1);
    const int num_tiles = AW / LCL_SZ;

    __local float Asub[LCL_SZ][LCL_SZ];
    __local float Bsub[LCL_SZ][LCL_SZ];
    float acc = 0.0f;

    for(int idx = 0; idx < num_tiles; idx++)
    {
        Asub[l_row][l_col] = A[g_row * AW + idx * LCL_SZ + l_col];
        Bsub[l_row][l_col] = B[(LCL_SZ * idx + l_row) * BW + g_col];

        barrier(CLK_LOCAL_MEM_FENCE);
        for(int lidx = 0; lidx < LCL_SZ; lidx++)
        {
            acc += Asub[l_row][lidx] * Bsub[lidx][l_col];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    C[g_row * BW + g_col] = acc;
}

