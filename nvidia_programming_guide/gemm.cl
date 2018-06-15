__kernel void gemm_naive(const __global float *A, const __global float *B, 
                         __global float *C)
{	          
    // Thread identifiers
    const int globalCol = get_global_id(0); // Row ID of C (0..M)
    const int globalRow = get_global_id(1); // Col ID of C (0..N)
				       
    // Compute a single element
    float acc = 0.0f;
    for (int k=0; k < AW; k++) 
    {
        acc += A[globalRow * AW + k] * B[k * BW + globalCol];
    }
    C[globalRow * AW + globalCol] = acc;
}

__kernel void gemm_local(const __global float *A, const __global float *B, 
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

__kernel void gemm_lclwpt(const __global float *A, const __global float *B, 
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

__kernel void gemm_lclvec(__global float *A, __global float *B, __global float *C)
{
    const int l_col = get_local_id(0);
    const int l_row = get_local_id(1);
    const int g_col = get_global_id(0) << 2;
    const int g_row = get_global_id(1);
    const int num_tiles = AW / LCL_SZ;
    
    __local float4 Asub[LCL_SZ][LCL_SZ >> 2];
    __local float4 Bsub[LCL_SZ][LCL_SZ >> 2];
    float4 acc = 0.0f;
    
    for(int idx = 0; idx < num_tiles; idx++)
    {
        Asub[l_row][l_col] = vload4(0, (__global float *)(A + g_row * AW + idx * LCL_SZ + (l_col << 2)));
	Bsub[l_row][l_col] = vload4(0, (__global float *)(B + (LCL_SZ * idx + l_row) * BW + g_col));
        barrier(CLK_LOCAL_MEM_FENCE);
	// LCL_SZ must be devided by 4
        for(int lidx = 0; lidx < LCL_SZ; lidx += 4)
	{
            acc += (float4)(Asub[l_row][lidx >> 2].s0) * Bsub[lidx + 0][l_col];
            acc += (float4)(Asub[l_row][lidx >> 2].s1) * Bsub[lidx + 1][l_col];
            acc += (float4)(Asub[l_row][lidx >> 2].s2) * Bsub[lidx + 2][l_col];
            acc += (float4)(Asub[l_row][lidx >> 2].s3) * Bsub[lidx + 3][l_col];
	} 
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    vstore4(acc, 0, (__global float *)(C + g_row * BW + g_col));
}

