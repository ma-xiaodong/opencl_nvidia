#include <iostream>
#include <sstream>
#include <fstream>
#include <CL/cl.h>
#include <math.h>
#include <iomanip>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include "gemm.hpp"

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

  commandQueue = clCreateCommandQueue(context, devices[0], 0, NULL);
  *device = devices[0];
  delete [] devices;

  return commandQueue;
}

cl_program CreateProgram(cl_context context, cl_device_id device,
                         const char *fileName)
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
  errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if(errNum != CL_SUCCESS)
  {
    char buildLog[16384];

    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                          sizeof(buildLog), buildLog, NULL);
    cerr << "Error in builing program: " << endl;
    cerr << buildLog << endl;
    clReleaseProgram(program);
    return NULL;
  }
  return program;
}

void Cleanup(cl_context context, cl_command_queue cmdQueue, 
             cl_program program, cl_kernel kernel,
             cl_mem *memObj0, cl_mem *memObj1, cl_mem *memObj2)
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

  if(memObj0 != NULL)
    clReleaseMemObject(*memObj0);

  if(memObj1 != NULL)
    clReleaseMemObject(*memObj1);

  if(memObj2 != NULL)
    clReleaseMemObject(*memObj2);

  return;
}

