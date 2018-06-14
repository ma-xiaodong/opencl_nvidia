#include <iostream>
#include <sstream>
#include <fstream>
#include <CL/cl.h>
#include <math.h>
#include <iomanip>
#include <sys/time.h>
#include "settings.h"

using namespace std;

void handle_error(cl_int ret, bool flag, const char *msg)
{
  if(ret || flag)
  {
    cout << msg << " error!" <<endl;
    exit(0);
  }
  return;
}

cl_context CreateContext()
{
  cl_int errNum;
  cl_uint numPlatforms;
  cl_platform_id firstPlatformId;
  cl_context context = NULL;

  errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
  handle_error(errNum != CL_SUCCESS, numPlatforms <= 0, "clGetPlatformIDs");

  cl_context_properties contextProperties[] = 
  {
    CL_CONTEXT_PLATFORM,
    (cl_context_properties)firstPlatformId,
    0
  };

  context = clCreateContextFromType(contextProperties, 
                                    CL_DEVICE_TYPE_GPU,
                                    NULL, NULL, &errNum);
  handle_error(errNum != CL_SUCCESS, context == NULL, "clCreateContextFromType");
  return context;
}

cl_command_queue CreateCommandQueue(cl_context context, cl_device_id *device)
{
  cl_int errNum;
  cl_device_id *devices;
  cl_command_queue commandQueue = NULL;
  size_t deviceBufferSize = -1;

  errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
  handle_error(errNum != CL_SUCCESS, false, "clGetContextInfo:0");
  handle_error(deviceBufferSize <= 0, false, "No device buffer!");
  cout << "deviceBufferSize: " << deviceBufferSize << endl;
  cout << "devices: " << deviceBufferSize / sizeof(cl_device_id)  << endl;

  devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
  errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize,
                            devices, NULL);
  handle_error(errNum != CL_SUCCESS, false, "clGetContextInfo:1");

  commandQueue = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, NULL);
  *device = devices[0];
  delete [] devices;

  return commandQueue;
}

cl_program CreateProgram(cl_context context, cl_device_id device,
                         const char *fileName, const char *build_opt)
{
  cl_int errNum;
  cl_program program;

  ifstream kernelFile(fileName, ios::in);
  if(!kernelFile.is_open())
  {
    cerr << "Failed to open: " << fileName << endl;
    return NULL;
  }

  ostringstream oss;
  oss << kernelFile.rdbuf();
  string srcStdStr = oss.str();
  const char *srcStr = srcStdStr.c_str();

  program = clCreateProgramWithSource(context, 1, (const char **)&srcStr, NULL, NULL);
  errNum = clBuildProgram(program, 0, NULL, build_opt, NULL, NULL);
  if(errNum != CL_SUCCESS)
  {
    char buildLog[16384];

    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                          sizeof(buildLog), buildLog, NULL);
    cerr << "Error in builing program: " << endl;
    cerr << buildLog << endl;

    kernelFile.close();
    clReleaseProgram(program);
    return NULL;
  }
  kernelFile.close();
  return program;
}

void Cleanup(cl_context context, cl_command_queue cmdQueue, 
             cl_program program, cl_kernel kernel,
             cl_mem *memObjects)
{
  int idx;

  if(context != NULL)
    clReleaseContext(context);

  if(cmdQueue != NULL)
    clReleaseCommandQueue(cmdQueue);

  if(program != NULL)
    clReleaseProgram(program);

  if(kernel != NULL)
    clReleaseKernel(kernel);

  for(idx = 0; idx < MEM_SIZE; idx++)
  {
    if(memObjects[idx] != NULL)
      clReleaseMemObject(memObjects[idx]);
  }
  return;
}

int compare_result(float *result, float *std_result, int size)
{
  // only compare a random point
  int flag = 1;
  for(int ii = 0; ii < size; ii++)
  {
    if((result[ii] - std_result[ii]) < -1e-2 || (result[ii] - std_result[ii]) > 1e-2)
    {
      flag = 0;
      cout << "Result error: [" << ii << "], ";
      cout << result[ii] << " : " << std_result[ii] << endl;
    }
  }

  return flag;
}

double timer(void)
{
  struct timeval Tvalue;
  struct timezone dummy;

  gettimeofday(&Tvalue, &dummy);
  double etime = (double)Tvalue.tv_sec + 1.0e-6*((double)Tvalue.tv_usec);

  return etime;
}

void tiled_mat_mul(float *a, float *b, float *result)
{
  int ah_blks = AH / TILE_SZ;
  int num_tiles = AW / TILE_SZ;
  int bw_blks = BW / TILE_SZ;

  for(int i = 0; i < ah_blks; i++)
  {
    for(int j = 0; j < bw_blks; j++)
    {
      for(int k = 0; k < num_tiles; k++)
      {
        for(int ii = 0; ii < TILE_SZ; ii++)
        {
          int a_m_idx = (i * TILE_SZ + ii) * AW + k * TILE_SZ;
          for(int jj = 0; jj < TILE_SZ; jj++)
          {
            int rlt_idx = (i * TILE_SZ + ii) * BW + j * TILE_SZ + jj;
            int b_m_idx = k * TILE_SZ * BW  + j * TILE_SZ + jj;

            for(int idx = 0; idx < TILE_SZ; idx++)
              result[rlt_idx] += a[a_m_idx + idx] * b[b_m_idx+ idx * BW];
          }
        }
      }
    }
  }
  return;
}

void get_perf_info(const char *msg, cl_event *event, bool flops)
{
  cl_ulong start_time, end_time;

  clGetEventProfilingInfo(*event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong),
                          &start_time, NULL);
  clGetEventProfilingInfo(*event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong),
                          &end_time, NULL);

  double total_time = (end_time - start_time) * 1e-6;
  printf("%s, time: %.2fms.", msg, total_time);
  if(flops)
  {
    double gflops;

    gflops = ((AW * 2) / total_time) * 1e-6 * AH * BW;
    printf(" gflops: %.2f.", gflops);
  }
  printf("\n");
  return;
}

int main(int argc, char **argv)
{
  // gemm on host
  float *a, *b;
  float *std_result, *result;
  int a_height, a_width, b_width;
  double s_time, e_time;

  a_height = AH;
  a_width = AW;
  b_width = BW;

  a = (float *)malloc(sizeof(float) * SIZE_A);
  b = (float *)malloc(sizeof(float) * SIZE_B);
  std_result = (float *)malloc(sizeof(float) * SIZE_C);
  result = (float *)malloc(sizeof(float) * SIZE_C);

  for(int idx = 0; idx < SIZE_A; idx++)
    a[idx] = rand() % 3 / 4.1;
  for(int idx = 0; idx < SIZE_B; idx++)
    b[idx] = rand() % 3 / 4.1;
  for(int idx = 0; idx < SIZE_C; idx++)
    std_result[idx] = 0.0f;

  s_time = timer();
  tiled_mat_mul(a, b, std_result);
  e_time = timer();
  cout << "Time used by tiled_mat_mul: " << e_time - s_time << endl;
  cout << "Gflops: " << AW * 2  / (e_time - s_time) * (BW / 1.0e9) * AH << endl;

  for(int idx = 0; idx < SIZE_C; idx++)
    result[idx] = 0.0f;

  // gemm on gpu
  cl_context context = 0;
  cl_command_queue commandQueue = 0;
  cl_program program = 0;
  cl_device_id device = 0;
  cl_kernel kernel = 0;
  cl_mem memObjects[MEM_SIZE] = {0, 0, 0};
  cl_int errNum;
  cl_event event = NULL;

  // create context
  context = CreateContext();

  // create command queue
  commandQueue = CreateCommandQueue(context, &device);
  if(commandQueue == NULL)
  {
    Cleanup(context, commandQueue, program, kernel, memObjects);
    return 1;
  }

  char build_opt[64];
  sprintf(build_opt, "%s%d %s%d %s%d %s%d", "-DAW=", AW, "-DBW=", BW, 
          "-DLCL_SZ=", LCL_SZ, "-DWPT=", WPT);

  // create opencl program for .cl kernel source
  program = CreateProgram(context, device, "gemm_naive.cl", build_opt);
  if(program == NULL)
  {
    Cleanup(context, commandQueue, program, kernel, memObjects);
    return 1;
  }

#ifdef LOCAL
  size_t globalWorkSize[2] = {BW, AH};
  size_t localWorkSize[2] = {LCL_SZ, LCL_SZ};
  kernel = clCreateKernel(program, "gemm_local", NULL);
#endif
#ifdef LCL_WPT
  size_t globalWorkSize[2] = {BW / WPT, AH};
  size_t localWorkSize[2] = {LCL_SZ / WPT, LCL_SZ};
  kernel = clCreateKernel(program, "gemm_lclwpt", NULL);
#endif

  if(kernel == NULL)
  {
    cerr << "Failed to create kernel!" << endl;
    Cleanup(context, commandQueue, program, kernel, memObjects);
    return 1;
  }

  memObjects[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 sizeof(float) * SIZE_A, a, NULL);
  memObjects[1] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 sizeof(float) * SIZE_B, b, NULL);
  memObjects[2] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                 sizeof(float) * SIZE_C, result, NULL);

  if(memObjects[0] == NULL || memObjects[1] == NULL ||
     memObjects[2] == NULL)
  {
    cerr << "Error creating memory objects!" << endl;
    Cleanup(context, commandQueue, program, kernel, memObjects);
    return 1;
  }
 
  errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &memObjects[0]);
  errNum = clSetKernelArg(kernel, 1, sizeof(cl_mem), &memObjects[1]);
  errNum = clSetKernelArg(kernel, 2, sizeof(cl_mem), &memObjects[2]);

  if(errNum != CL_SUCCESS)
  {
    cerr << "Error setting kernel arguments!"  << endl;
    Cleanup(context, commandQueue, program, kernel, memObjects);
    return 1;
  }

  errNum = clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, 
                                  globalWorkSize, localWorkSize,
                                  0, NULL, &event);
  
  errNum = clWaitForEvents(1, &event);
  get_perf_info("Opencl", &event, 1);
  if(errNum != CL_SUCCESS)
  {
    cerr << "clWaitForEvents" << endl;
    Cleanup(context, commandQueue, program, kernel, memObjects);
    return 1;
  }

  errNum = clEnqueueReadBuffer(commandQueue, memObjects[2],
                               CL_TRUE, 0, SIZE_C * sizeof(float),
                               result, 0, NULL, &event);
  
  errNum = clWaitForEvents(1, &event);
  if(errNum != CL_SUCCESS)
  {
    cerr << "clEnqueueReadBuffer"  << endl;
    Cleanup(context, commandQueue, program, kernel, memObjects);
    return 1;
  }
  cout << "Executed program successfully." << endl;

  if(!compare_result(result, std_result, SIZE_C))
    cout << "The result of opencl is wrong!" << endl;
  else
    cout << "The result of opencl is correct!" << endl;

  Cleanup(context, commandQueue, program, kernel, memObjects);

  free(a);
  free(b);
  free(result);
  free(std_result);

  return 0;
}

