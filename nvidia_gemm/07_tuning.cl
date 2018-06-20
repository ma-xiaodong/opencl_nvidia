#pragma OPENCL EXTENSION cl_amd_printf : enable

__kernel void tuning4x4_opt(__global float *A, __global float *B, __global float *C)
{
    const int l_col = get_local_id(0) << 2;
    const int l_row = get_local_id(1) << 2;
    const int g_col = get_global_id(0) << 2;
    const int g_row = get_global_id(1) << 2;
    const int num_tiles = AW / TILE_K;

   // Rectangular local matrix instead of square
    __local float Asub[TILE_MN][TILE_K];
    __local float Bsub[TILE_K][TILE_MN];
    float4 acc[4], bReg[4], aReg;

    acc[0] = (float4)0.0f; acc[1] = (float4)0.0f;
    acc[2] = (float4)0.0f; acc[3] = (float4)0.0f;

    // how many array elements should each work item load?
    unsigned int eles_4_wi = (TILE_MN * TILE_K) / ((TILE_MN >> 2) * (TILE_MN >> 2));
    // the element current work item start to load
    unsigned int start_ele = (l_row >> 2) * (TILE_MN >> 2) + (l_col >> 2);
    // load step
    unsigned int load_step = (TILE_MN >> 2) * (TILE_MN >> 2);
    
    for(int ii = 0; ii < num_tiles; ii++)
    {
	for(int jj = 0; jj < eles_4_wi; jj++)
	{
	    int a_row = (start_ele + jj * load_step) / TILE_K;
	    int a_col = (start_ele + jj * load_step) % TILE_K;
            Asub[a_row][a_col] = A[(get_group_id(1) * TILE_MN + a_row) * AW + ii * TILE_K + a_col];
	    Bsub[a_col][a_row] = B[(ii * TILE_K + a_col) * BW + get_group_id(0) * TILE_MN + a_row];
	}
        barrier(CLK_LOCAL_MEM_FENCE);

	// TILE_MN must be devided by 4
        for(int lidx = 0; lidx < TILE_K; lidx += 4)
	{
	    bReg[0] = (float4)(Bsub[lidx][l_col], Bsub[lidx][l_col + 1], Bsub[lidx][l_col + 2], Bsub[lidx][l_col + 3]);
	    bReg[1] = (float4)(Bsub[lidx + 1][l_col], Bsub[lidx + 1][l_col + 1], Bsub[lidx + 1][l_col + 2], Bsub[lidx + 1][l_col + 3]);
	    bReg[2] = (float4)(Bsub[lidx + 2][l_col], Bsub[lidx + 2][l_col + 1], Bsub[lidx + 2][l_col + 2], Bsub[lidx + 2][l_col + 3]);
	    bReg[3] = (float4)(Bsub[lidx + 3][l_col], Bsub[lidx + 3][l_col + 1], Bsub[lidx + 3][l_col + 2], Bsub[lidx + 3][l_col + 3]);

	    aReg = (float4)(Asub[l_row][lidx], Asub[l_row][lidx + 1], Asub[l_row][lidx + 2], Asub[l_row][lidx + 3]);
            acc[0] += (float4)(aReg.s0) * bReg[0];
            acc[0] += (float4)(aReg.s1) * bReg[1];
            acc[0] += (float4)(aReg.s2) * bReg[2];
            acc[0] += (float4)(aReg.s3) * bReg[3];

	    aReg = (float4)(Asub[l_row + 1][lidx], Asub[l_row + 1][lidx + 1], Asub[l_row + 1][lidx + 2], Asub[l_row + 1][lidx + 3]);
            acc[1] += (float4)(aReg.s0) * bReg[0];
            acc[1] += (float4)(aReg.s1) * bReg[1];
            acc[1] += (float4)(aReg.s2) * bReg[2];
            acc[1] += (float4)(aReg.s3) * bReg[3];

	    aReg = (float4)(Asub[l_row + 2][lidx], Asub[l_row + 2][lidx + 1], Asub[l_row + 2][lidx + 2], Asub[l_row + 2][lidx + 3]);
            acc[2] += (float4)(aReg.s0) * bReg[0];
            acc[2] += (float4)(aReg.s1) * bReg[1];
            acc[2] += (float4)(aReg.s2) * bReg[2];
            acc[2] += (float4)(aReg.s3) * bReg[3];

	    aReg = (float4)(Asub[l_row + 3][lidx], Asub[l_row + 3][lidx + 1], Asub[l_row + 3][lidx + 2], Asub[l_row + 3][lidx + 3]);
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

