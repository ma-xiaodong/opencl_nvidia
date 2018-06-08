#pragma OPENCL EXTENSION cl_amd_printf : enable
__kernel void float_add(__global float *input)
{
  int gid = (int)get_global_id(0);
  printf("gid: %d\n", gid);
  input[gid] = gid + 0.1;
}

