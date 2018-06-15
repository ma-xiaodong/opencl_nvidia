__kernel void transpose(__global float *B, __global float *TB)
{
    const int l_col = get_local_id(0);
    const int l_row = get_local_id(1);
    const int g_col = get_global_id(0);
    const int g_row = get_global_id(1);

    TB[g_col * BW + g_row] = B[g_row * BW + g_col];

    return;
}

__kernel void lclvec_trans_opt(__global float *A, __global float *TB, __global float *C)
{
    return;
}
