#include "settings.h"

__kernel void rcl__kernel0(__global float* A, __global float* B, __global float* C) {
  int gid = (int)get_group_id(0);
  int lid = (int)get_local_id(0);

  for (int y = 0; y < BW; y++) {
    C[(gid * TS + lid) * BW + y] = 0.0f;

    for (int k = 0; k < AW; k++) {
      C[(gid * TS + lid) * BW + y] += A[(gid * TS + lid) * AW + k] * B[y + k * BW];
    }
  }
}
