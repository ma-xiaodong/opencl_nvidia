#include <stdio.h>
#include <CL/cl.h>

#define VECTOR_SIZE	1024

const char *saxpy_kernel = 
"__kernel					\n"
"void saxpy_kernel(float alpha,			\n"
"                  __global float *A,		\n"
"                  __global float *B,		\n"
"                  __global float *C)		\n"
"						\n"
"{						\n"
"   int index = get_global_id(0);		\n"
"   C[index] = alpha * A[index] + B[index];	\n"
"}						\n";

double get_exec_time(cl_event *event)
{
  cl_ulong start_time, end_time;

  clGetEventProfilingInfo(*event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong),
                          &start_time, NULL);
  clGetEventProfilingInfo(*event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong),
                          &end_time, NULL);
  double total_time = (end_time - start_time) * 1e-6;
  return total_time;
}

int main(void)
{
  int i;

  float alpha = 2.0;
  float *A = (float*)malloc(sizeof(float) * VECTOR_SIZE);
  float *B = (float*)malloc(sizeof(float) * VECTOR_SIZE);
  float *C = (float*)malloc(sizeof(float) * VECTOR_SIZE);

  for(i = 0; i < VECTOR_SIZE; i++)
  {
    A[i] = i;
    B[i] = VECTOR_SIZE;
    C[i] = 0;
  }

  cl_platform_id *platforms = NULL;
  cl_uint num_platforms;
  cl_int cl_status = clGetPlatformIDs(0, NULL, &num_platforms);
  platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * num_platforms);
  cl_status = clGetPlatformIDs(num_platforms, platforms, NULL);

  cl_device_id *device_list = NULL;
  cl_uint num_devices;

  cl_event event;

  cl_status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
  device_list = (cl_device_id *)malloc(sizeof(cl_device_id) * num_devices);
  cl_status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, num_devices,
                             device_list, NULL);
  cl_context context;
  context = clCreateContext(NULL, num_devices, device_list, NULL, NULL, &cl_status);
  cl_command_queue command_queue = clCreateCommandQueue(context, device_list[0],
                                   CL_QUEUE_PROFILING_ENABLE, &cl_status);

  cl_mem A_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY, 
                                  VECTOR_SIZE * sizeof(float), NULL, &cl_status);
  cl_mem B_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY, 
                                  VECTOR_SIZE * sizeof(float), NULL, &cl_status);
  cl_mem C_clmem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
                                  VECTOR_SIZE * sizeof(float), NULL, &cl_status);

  cl_status = clEnqueueWriteBuffer(command_queue, A_clmem, CL_TRUE, 0, VECTOR_SIZE *
                                   sizeof(float), A, 0, NULL, &event);
  clWaitForEvents(1, &event);
  printf("time used by write A: %fms.\n", get_exec_time(&event));

  cl_status = clEnqueueWriteBuffer(command_queue, B_clmem, CL_TRUE, 0, VECTOR_SIZE *
                                   sizeof(float), B, 0, NULL, &event);
  clWaitForEvents(1, &event);
  printf("time used by write B: %fms.\n", get_exec_time(&event));

  cl_program program = clCreateProgramWithSource(context, 1, (const char **)&saxpy_kernel,
                                                 NULL, &cl_status);
  cl_status = clBuildProgram(program, 1, device_list, NULL, NULL, NULL);

  cl_kernel kernel = clCreateKernel(program, "saxpy_kernel", &cl_status);

  cl_status = clSetKernelArg(kernel, 0, sizeof(float), (void *)&alpha);
  cl_status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&A_clmem);
  cl_status = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&B_clmem);
  cl_status = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&C_clmem);

  size_t global_size = VECTOR_SIZE;
  size_t local_size = 64;

  cl_status = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_size,
                                     &local_size, 0, NULL, &event);
  clWaitForEvents(1, &event);
  printf("time used by the main command: %fms.\n", get_exec_time(&event));
  cl_status = clEnqueueReadBuffer(command_queue, C_clmem, CL_TRUE, 0, VECTOR_SIZE *
                                  sizeof(float), C, 0, NULL, NULL);
  cl_status = clFlush(command_queue);
  cl_status = clFinish(command_queue);

/*
  for(i = 0; i < VECTOR_SIZE; i++)
  {
    printf("%f * %f + %f = %f\n", alpha, A[i], B[i], C[i]);
  }
*/

  cl_status = clReleaseEvent(event);
  cl_status = clReleaseKernel(kernel);
  cl_status = clReleaseProgram(program);
  cl_status = clReleaseMemObject(A_clmem);
  cl_status = clReleaseMemObject(B_clmem);
  cl_status = clReleaseMemObject(C_clmem);
  cl_status = clReleaseCommandQueue(command_queue);
  cl_status = clReleaseContext(context);

  free(A);
  free(B);
  free(C);
  free(platforms);
  free(device_list);
  return 0;
}

