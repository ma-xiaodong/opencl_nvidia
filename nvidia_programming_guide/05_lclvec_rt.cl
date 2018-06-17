__kernel void lclvec_rt_opt(__global float *A, __global float *B, __global float *C)
{
    const int l_col = get_local_id(0);
    const int l_row = get_local_id(1) << 2;
    const int g_col = get_global_id(0) << 2;
    const int g_row = get_global_id(1) << 2;
    const int num_tiles = AW / LCL_SZ;

    __local float4 Asub[LCL_SZ][LCL_SZ >> 2];
    __local float4 Bsub[LCL_SZ][LCL_SZ >> 2];
    float4 acc[4], bReg[4], aReg;

    acc[0] = (float4)0.0f; acc[1] = (float4)0.0f;
    acc[2] = (float4)0.0f; acc[3] = (float4)0.0f;
    
    for(int idx = 0; idx < num_tiles; idx++)
    {
        Asub[l_row][l_col] = vload4(0, (__global float *)(A + g_row * AW + idx * LCL_SZ + (l_col << 2)));
        Asub[l_row + 1][l_col] = vload4(0, (__global float *)(A + (g_row + 1) * AW + idx * LCL_SZ + (l_col << 2)));
        Asub[l_row + 2][l_col] = vload4(0, (__global float *)(A + (g_row + 2) * AW + idx * LCL_SZ + (l_col << 2)));
        Asub[l_row + 3][l_col] = vload4(0, (__global float *)(A + (g_row + 3) * AW + idx * LCL_SZ + (l_col << 2)));

	Bsub[l_row][l_col] = vload4(0, (__global float *)(B + (LCL_SZ * idx + l_row) * BW + g_col));
	Bsub[l_row + 1][l_col] = vload4(0, (__global float *)(B + (LCL_SZ * idx + l_row + 1) * BW + g_col));
	Bsub[l_row + 2][l_col] = vload4(0, (__global float *)(B + (LCL_SZ * idx + l_row + 2) * BW + g_col));
	Bsub[l_row + 3][l_col] = vload4(0, (__global float *)(B + (LCL_SZ * idx + l_row + 3) * BW + g_col));
        barrier(CLK_LOCAL_MEM_FENCE);

	// LCL_SZ must be devided by 4
        for(int lidx = 0; lidx < LCL_SZ; lidx += 4)
	{
	    bReg[0] = Bsub[lidx][l_col];
	    bReg[1] = Bsub[lidx + 1][l_col];
	    bReg[2] = Bsub[lidx + 2][l_col];
	    bReg[3] = Bsub[lidx + 3][l_col];

	    aReg = Asub[l_row][lidx >> 2];
            acc[0] += (float4)(aReg.s0) * bReg[0];
            acc[0] += (float4)(aReg.s1) * bReg[1];
            acc[0] += (float4)(aReg.s2) * bReg[2];
            acc[0] += (float4)(aReg.s3) * bReg[3];

	    aReg = Asub[l_row + 1][lidx >> 2];
            acc[1] += (float4)(aReg.s0) * bReg[0];
            acc[1] += (float4)(aReg.s1) * bReg[1];
            acc[1] += (float4)(aReg.s2) * bReg[2];
            acc[1] += (float4)(aReg.s3) * bReg[3];

	    aReg = Asub[l_row + 2][lidx >> 2];
            acc[2] += (float4)(aReg.s0) * bReg[0];
            acc[2] += (float4)(aReg.s1) * bReg[1];
            acc[2] += (float4)(aReg.s2) * bReg[2];
            acc[2] += (float4)(aReg.s3) * bReg[3];

	    aReg = Asub[l_row + 3][lidx >> 2];
            acc[3] += (float4)(aReg.s0) * bReg[0];
            acc[3] += (float4)(aReg.s1) * bReg[1];
            acc[3] += (float4)(aReg.s2) * bReg[2];
            acc[3] += (float4)(aReg.s3) * bReg[3];
	} 
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    vstore4(acc[0], 0, (__global float *)(C + g_row * BW + g_col));
    vstore4(acc[1], 0, (__global float *)(C + (g_row + 1) * BW + g_col));
    vstore4(acc[2], 0, (__global float *)(C + (g_row + 2) * BW + g_col));
    vstore4(acc[3], 0, (__global float *)(C + (g_row + 3) * BW + g_col));
    return;
}

