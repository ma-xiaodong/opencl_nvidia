__kernel void lclwpt_opt(const __global float *A, const __global float *B, 
                         __global float *C)
{
    const int l_col = get_local_id(0) * WPT;
    const int l_row = get_local_id(1);
    const int g_col = get_global_id(0) * WPT;
    const int g_row = get_global_id(1);
    const int num_tiles = AW / LCL_SZ;

    __local float Asub[LCL_SZ][LCL_SZ];
    __local float Bsub[LCL_SZ][LCL_SZ];
    float acc[WPT];

    for(int idx = 0; idx < WPT; idx++)
    {
	acc[idx] = 0.0f;        
    }

    for(int idx = 0; idx < num_tiles; idx++)
    {
        for(int w = 0; w < WPT; w++)
	{
            Asub[l_row][l_col + w] = A[g_row * AW + idx * LCL_SZ + l_col + w];
            Bsub[l_row][l_col + w] = B[(LCL_SZ * idx + l_row) * BW + g_col + w];
	}

        barrier(CLK_LOCAL_MEM_FENCE);
        for(int lidx = 0; lidx < LCL_SZ; lidx++)
        {
	    for(int w = 0; w < WPT; w++)
	    {
		acc[w] += Asub[l_row][lidx] * Bsub[lidx][l_col + w];
	    }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for(int w = 0; w < WPT; w++)
	C[g_row * BW + g_col + w] = acc[w];
}

