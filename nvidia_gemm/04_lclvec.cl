__kernel void lclvec_opt(__global float *A, __global float *B, __global float *C)
{
    const int l_col = get_local_id(0);
    const int l_row = get_local_id(1);
    const int g_col = get_global_id(0) << 2;
    const int g_row = get_global_id(1);
    const int num_tiles = AW / LCL_SZ;
    
    __local float4 Asub[LCL_SZ][LCL_SZ >> 2];
    __local float4 Bsub[LCL_SZ][LCL_SZ >> 2];
    float4 acc = (float4)0.0f, aReg = (float4)0.0f;
    
    for(int idx = 0; idx < num_tiles; idx++)
    {
        Asub[l_row][l_col] = vload4(0, (__global float *)(A + g_row * AW + idx * LCL_SZ + (l_col << 2)));
	Bsub[l_row][l_col] = vload4(0, (__global float *)(B + (LCL_SZ * idx + l_row) * BW + g_col));
        barrier(CLK_LOCAL_MEM_FENCE);

	// LCL_SZ must be devided by 4
        for(int lidx = 0; lidx < LCL_SZ; lidx += 4)
	{
	    aReg = Asub[l_row][lidx >> 2];
            acc += (float4)(aReg.s0) * Bsub[lidx + 0][l_col];
            acc += (float4)(aReg.s1) * Bsub[lidx + 1][l_col];
            acc += (float4)(aReg.s2) * Bsub[lidx + 2][l_col];
            acc += (float4)(aReg.s3) * Bsub[lidx + 3][l_col];
	} 
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    vstore4(acc, 0, (__global float *)(C + g_row * BW + g_col));
}

