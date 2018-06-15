__kernel void naive(const __global float *A, const __global float *B, 
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

