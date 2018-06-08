#include <stdio.h>
#include <CL/cl.h>
#include <string.h>
#include <stdlib.h>
#include "util.h"

extern cl_platform_id __platform;
extern cl_device_id __device;
extern cl_context __context;
extern cl_command_queue __command_queue;

float *a, *b, *c;

void compare_result(float *a, float *b, float *result)
{
  // only compare a random point
  int r_row = rand() % AH;
  int r_col = rand() % BW;
  float t_val = 0.0f;

  for(int idx = 0; idx < AW; idx++)
  {
    t_val += a[r_row * AW + idx] * b[idx * BW + r_col];
  }

  if(t_val != result[r_row * BW + r_col])
    printf("Result error: [%d, %d], %f:%f.\n", r_row, r_col, result[r_row * BW + r_col], t_val);
  else
    printf("Result correct!\n");

  return;
}

// parameter indicates whether use local optimization
void twodim_tune(bool local)
{
  cl_int cl_status;
  cl_program program;
  cl_kernel kernel;
  cl_mem mem0, mem1, mem2;
  cl_event event;
  size_t global_size[2], local_size[2];
  char build_opt[64] = "\0";

  sprintf(build_opt, "%s%d %s%d %s%d %s%d", "-DAH=", AH, "-DBW=", BW, "-DAW=", AW, "-DBLK=", BLK);
  strcat(build_opt, "\0");

  program = CreateProgram(__context, __device, "tune_kernel.cl", build_opt);
  if(local)
    kernel = clCreateKernel(program, "gemm_local", &cl_status);
  else
    kernel = clCreateKernel(program, "gemm_origin", &cl_status);
  if(cl_status != CL_SUCCESS)
  {
    printf("Error: clCreateKernel!\n");
    clReleaseProgram(program);
    return;
  }

  mem0 = clCreateBuffer(__context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 
                        sizeof(float) * AH * AW, a, &cl_status);
  mem1 = clCreateBuffer(__context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 
                        sizeof(float) * BH * BW, b, &cl_status);
  mem2 = clCreateBuffer(__context, CL_MEM_WRITE_ONLY, 
                        sizeof(float) * AH * BW, NULL, &cl_status);

  clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&mem0);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&mem1);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&mem2);

  global_size[0] = BW; global_size[1] = AH;
  local_size[0] = BLK; local_size[1] = BLK;

  cl_status = clEnqueueNDRangeKernel(__command_queue, kernel, 2, NULL, global_size, 
                                     local_size, 0, NULL, &event);

  clWaitForEvents(1, &event);
  if(local)
    get_perf_info("twodim run local", &event, true);
  else
    get_perf_info("twodim run origin", &event, true);


  cl_status = clEnqueueReadBuffer(__command_queue, mem2, CL_FALSE, 0, 
                                  sizeof(float) * AH * BW, c, 0, NULL, &event);
  clWaitForEvents(1, &event);
  if(local)
    get_perf_info("twodim local ReadBuffer", &event, false);
  else
    get_perf_info("twodim origin ReadBuffer", &event, false);

  compare_result(a, b, c);

  clReleaseEvent(event);
  clReleaseMemObject(mem0);
  clReleaseMemObject(mem1);
  clReleaseMemObject(mem2);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
}

void onerow_tune()
{
  cl_int cl_status;
  cl_program program;
  cl_kernel kernel;
  cl_mem mem0, mem1, mem2;
  cl_event event;

  size_t global_size, local_size;
  char build_opt[64];

  sprintf(build_opt, "%s%d %s%d %s%d %s%d", "-DAH=", AH, "-DBW=", BW, "-DAW=", AW, "-DBLK=", BLK);
  strcat(build_opt, "\0");
  program = CreateProgram(__context, __device, "tune_kernel.cl", build_opt);
  kernel = clCreateKernel(program, "gemm_onerow", &cl_status);

  if(cl_status != CL_SUCCESS)
  {
    printf("Error: clCreateKernel!\n");
    clReleaseProgram(program);
    return;
  }

  mem0 = clCreateBuffer(__context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 
                        sizeof(float) * AH * AW, a, &cl_status);
  mem1 = clCreateBuffer(__context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 
                        sizeof(float) * BH * BW, b, &cl_status);
  mem2 = clCreateBuffer(__context, CL_MEM_WRITE_ONLY, 
                        sizeof(float) * AH * BW, NULL, &cl_status);

  clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&mem0);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&mem1);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&mem2);

  global_size = AH; local_size = BLK;

  cl_status = clEnqueueNDRangeKernel(__command_queue, kernel, 1, NULL, &global_size, 
                                     &local_size, 0, NULL, &event);

  clWaitForEvents(1, &event);
  get_perf_info("onerow run", &event, true);

  cl_status = clEnqueueReadBuffer(__command_queue, mem2, CL_FALSE, 0, 
                                  sizeof(float) * AH * BW, c, 0, NULL, &event);
  clWaitForEvents(1, &event);
  get_perf_info("onerow Readbuffer", &event, false);
  compare_result(a, b, c);

  clReleaseEvent(event);
  clReleaseMemObject(mem0);
  clReleaseMemObject(mem1);
  clReleaseMemObject(mem2);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
}

void copy_host()
{
  cl_int cl_status;
  cl_program program;
  cl_kernel kernel;
  cl_mem mem0, mem1, mem2;
  cl_event event;
  size_t global_size[2], local_size[2];
  char build_opt[64] = "\0";

  sprintf(build_opt, "%s%d %s%d %s%d %s%d", "-DAH=", AH, "-DBW=", BW, "-DAW=", AW, "-DBLK=", BLK);
  strcat(build_opt, "\0");

  program = CreateProgram(__context, __device, "tune_kernel.cl", build_opt);
  kernel = clCreateKernel(program, "gemm_origin", &cl_status);

  if(cl_status != CL_SUCCESS)
  {
    printf("Error: clCreateKernel!\n");
    clReleaseProgram(program);
    return;
  }

  mem0 = clCreateBuffer(__context, CL_MEM_COPY_HOST_PTR, 
                        sizeof(float) * AH * AW, a, &cl_status);
  mem1 = clCreateBuffer(__context, CL_MEM_COPY_HOST_PTR, 
                        sizeof(float) * BH * BW, b, &cl_status);
  mem2 = clCreateBuffer(__context, CL_MEM_WRITE_ONLY, 
                        sizeof(float) * AH * BW, NULL, &cl_status);

  clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&mem0);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&mem1);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&mem2);

  global_size[0] = BW; global_size[1] = AH;
  local_size[0] = BLK; local_size[1] = BLK;

  cl_status = clEnqueueNDRangeKernel(__command_queue, kernel, 2, NULL, global_size, 
                                     local_size, 0, NULL, &event);

  clWaitForEvents(1, &event);
  get_perf_info("copy host", &event, true);

  cl_status = clEnqueueReadBuffer(__command_queue, mem2, CL_FALSE, 0, 
                                  sizeof(float) * AH * BW, c, 0, NULL, &event);
  clWaitForEvents(1, &event);
  get_perf_info("copy host", &event, false);
  compare_result(a, b, c);

  clReleaseEvent(event);
  clReleaseMemObject(mem0);
  clReleaseMemObject(mem1);
  clReleaseMemObject(mem2);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
}

void alloc_host()
{
  cl_int cl_status;
  cl_program program;
  cl_kernel kernel;
  cl_mem mem0, mem1, mem2;
  cl_event event;
  size_t global_size[2], local_size[2];
  char build_opt[64] = "\0";
  float *tmpa, *tmpb, *tmpc;

  sprintf(build_opt, "%s%d %s%d %s%d %s%d", "-DAH=", AH, "-DBW=", BW, "-DAW=", AW, "-DBLK=", BLK);
  strcat(build_opt, "\0");

  program = CreateProgram(__context, __device, "tune_kernel.cl", build_opt);
  kernel = clCreateKernel(program, "gemm_origin", &cl_status);

  if(cl_status != CL_SUCCESS)
  {
    printf("Error: clCreateKernel!\n");
    clReleaseProgram(program);
    return;
  }

  // use ALLOC_HOST_PTR to alloc memory on the device
  mem0 = clCreateBuffer(__context, CL_MEM_ALLOC_HOST_PTR, 
                        sizeof(float) * AH * AW, NULL, &cl_status);
  mem1 = clCreateBuffer(__context, CL_MEM_ALLOC_HOST_PTR, 
                        sizeof(float) * BH * BW, NULL, &cl_status);
  mem2 = clCreateBuffer(__context, CL_MEM_ALLOC_HOST_PTR,
                        sizeof(float) * AH * BW, NULL, &cl_status);

  // map the mem0 on device to host and initialize it
  tmpa = (float *)clEnqueueMapBuffer(__command_queue, mem0, CL_FALSE, CL_MAP_WRITE, 0,
                                     sizeof(float) * AH * AW, 0, NULL, &event, &cl_status);
  clWaitForEvents(1, &event);
  get_perf_info("alloc host map mem0:", &event, false);
  for(int idx = 0; idx < AH * AW; idx++)
    tmpa[idx] = rand() % 3;

  clEnqueueUnmapMemObject(__command_queue, mem0, tmpa, 0, NULL, &event);
  clWaitForEvents(1, &event);
  get_perf_info("alloc host unmap mem0:", &event, false);

  // map the mem1 on device to host and initialize it
  tmpb = (float *)clEnqueueMapBuffer(__command_queue, mem1, CL_FALSE, CL_MAP_WRITE, 0,
                              sizeof(float) * BH * BW, 0, NULL, &event, &cl_status);
  clWaitForEvents(1, &event);
  get_perf_info("alloc host map mem1:", &event, false);
  for(int idx = 0; idx < AH * AW; idx++)
    tmpb[idx] = rand() % 3;
  clEnqueueUnmapMemObject(__command_queue, mem1, tmpb, 0, NULL, &event);
  clWaitForEvents(1, &event);
  get_perf_info("alloc host unmap mem1:", &event, false);

  clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&mem0);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&mem1);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&mem2);

  global_size[0] = BW; global_size[1] = AH;
  local_size[0] = BLK; local_size[1] = BLK;

  cl_status = clEnqueueNDRangeKernel(__command_queue, kernel, 2, NULL, global_size, 
                                     local_size, 0, NULL, &event);

  clWaitForEvents(1, &event);
  get_perf_info("alloc kernel exec time", &event, true);

  tmpa = (float *)clEnqueueMapBuffer(__command_queue, mem0, CL_FALSE, CL_MAP_WRITE, 0,
                              sizeof(float) * BH * BW, 0, NULL, &event, &cl_status);
  clWaitForEvents(1, &event);
  get_perf_info("alloc host map mem0:", &event, false);

  tmpb = (float *)clEnqueueMapBuffer(__command_queue, mem1, CL_FALSE, CL_MAP_WRITE, 0,
                              sizeof(float) * BH * BW, 0, NULL, &event, &cl_status);
  clWaitForEvents(1, &event);
  get_perf_info("alloc host map mem1:", &event, false);

  tmpc = (float *)clEnqueueMapBuffer(__command_queue, mem2, CL_FALSE, CL_MAP_WRITE, 0,
                              sizeof(float) * BH * BW, 0, NULL, &event, &cl_status);
  clWaitForEvents(1, &event);
  get_perf_info("alloc host map mem2:", &event, false);

  compare_result(tmpa, tmpb, tmpc);

  clEnqueueUnmapMemObject(__command_queue, mem0, tmpa, 0, NULL, &event);
  clWaitForEvents(1, &event);
  get_perf_info("alloc host unmap mem0:", &event, false);

  clEnqueueUnmapMemObject(__command_queue, mem1, tmpb, 0, NULL, &event);
  clWaitForEvents(1, &event);
  get_perf_info("alloc host unmap mem1:", &event, false);

  clEnqueueUnmapMemObject(__command_queue, mem2, tmpc, 0, NULL, &event);
  clWaitForEvents(1, &event);
  get_perf_info("alloc host unmap mem2:", &event, false);

  clReleaseEvent(event);
  clReleaseMemObject(mem0);
  clReleaseMemObject(mem1);
  clReleaseMemObject(mem2);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
}

int main(int argc, char **argv)
{
  setup_cl();
  a = (float *)malloc(sizeof(float) * AH * AW);
  b = (float *)malloc(sizeof(float) * BH * BW);
  c = (float *)malloc(sizeof(float) * AH * BW);

  for(int idx = 0; idx < AH * AW; idx++)
    a[idx] = rand() % 3;
  for(int idx = 0; idx < BH * BW; idx++)
    b[idx] = rand() % 3;
  for(int idx = 0; idx < AH * BW; idx++)
    c[idx] = 0.0f;

  // first call
  twodim_tune(false);

  // second call
  for(int idx = 0; idx < AH * BW; idx++)
    c[idx] = 0.0f;
  onerow_tune();

  // third call
  for(int idx = 0; idx < AH * BW; idx++)
    c[idx] = 0.0f;
  twodim_tune(true);

  // forth call
  for(int idx = 0; idx < AH * BW; idx++)
    c[idx] = 0.0f;
  copy_host();

  // fifth call
  for(int idx = 0; idx < AH * BW; idx++)
    c[idx] = 0.0f;
  alloc_host();

  free(a);
  free(b);
  free(c);
  clean_cl();
}
