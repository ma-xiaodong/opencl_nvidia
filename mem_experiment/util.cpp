#include <iostream>
#include <sstream>
#include <fstream>
#include <stdio.h>
#include <CL/cl.h>
#include "util.h"

using namespace std;

// cl related variables
cl_platform_id __platform;
cl_device_id __device;
cl_context __context;
cl_command_queue __command_queue;

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

void setup_cl()
{
  cl_uint num_platforms, num_devices;
  cl_int cl_status;

  // platforms
  cl_status = clGetPlatformIDs(0, NULL, &num_platforms);
  
  if(cl_status != CL_SUCCESS)
  {
    printf("Error: clGetPlatformIDs!\n");
    return;
  }
  else if(num_platforms < 1)
  {
    printf("Error: no platform found!\n");
    return;
  }
  else
    printf("Number of platforms: %d.\n", num_platforms);

  cl_status = clGetPlatformIDs(1, &__platform, NULL);
  if(cl_status != CL_SUCCESS)
  {
    printf("Error: second call of clGetPlatformIDs!\n");
    return;
  }

  // devices
  cl_status = clGetDeviceIDs(__platform, CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
  if(cl_status != CL_SUCCESS)
  {
    printf("Error: clGetDeviceIDs!\n");
    return;
  }
  else if(num_devices < 1)
  {
    printf("Error: no devices found!\n");
    return;
  }
  else
    printf("Number of devices: %d.\n", num_devices);

  cl_status = clGetDeviceIDs(__platform, CL_DEVICE_TYPE_GPU, 1, &__device, NULL);
  if(cl_status != CL_SUCCESS)
  {
    printf("Error: second call of clGetDeviceIDs!\n");
    return;
  }

  // context
  __context = clCreateContext(NULL, 1, &__device, NULL, NULL, &cl_status);
  __command_queue = clCreateCommandQueue(__context, __device, CL_QUEUE_PROFILING_ENABLE,
                                         &cl_status);
  return;
}

cl_program CreateProgram(cl_context context, cl_device_id device,
                         const char *fileName, const char *build_opt)
{
    cl_int errNum;
    cl_program program;

    ifstream kernelFile(fileName, ios::in);
    if(!kernelFile.is_open())
    {
      printf("Failed to open: %s!\n", fileName);
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

      clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buildLog), 
	                    buildLog, NULL);
      printf("Error in builing program: \n");
      printf("%s\n", buildLog);
      clReleaseProgram(program);
      kernelFile.close();
      return NULL;
    }
    kernelFile.close();
    return program;
}

void clean_cl()
{
  clReleaseCommandQueue(__command_queue);
  clReleaseContext(__context);
  return;
}
