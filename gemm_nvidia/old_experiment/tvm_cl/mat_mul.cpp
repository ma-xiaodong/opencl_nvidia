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

int compare_result(float *a, float *b, float *result)
{
  // only compare a random point
  int r_row = rand() % AH;
  int r_col = rand() % BW;
  float t_val = 0.0f;

  for(int j = 0; j < AW; j++)
  {
    t_val += a[r_row * AW + j] * b[j * BW + r_col];
  }

  if(t_val != result[r_row * BW + r_col])
  {
    cout << "Result error: [" << r_row << ", " << r_col << "], ";
    cout << result[r_row * BW + r_col] << " : " << t_val << endl;
    return 0;
  }

  return 1;
}

double timer(void)
{
  struct timeval Tvalue;
  struct timezone dummy;

  gettimeofday(&Tvalue, &dummy);
  double etime = (double)Tvalue.tv_sec + 1.0e-6*((double)Tvalue.tv_usec);

  return etime;
}

void mat_mul(float *a, float *b, float *result)
{
  for(int i = 0; i < AH; i++)
  {
    int a_m_idx = i * AW;
    for(int j = 0; j < BW; j++)
    {
      int rlt_m_idx = i * BW + j;
      for(int k = 0; k < AW; k++)
        result[rlt_m_idx] += a[a_m_idx + k] * b[k * BW + j];
    }
  }
}

void tiled_mat_mul(float *a, float *b, float *result)
{
  int ah_blks = AH / TS;
  int num_tiles = AW / TS;
  int bw_blks = BW / TS;

  for(int i = 0; i < ah_blks; i++)
  {
    for(int j = 0; j < bw_blks; j++)
    {
      for(int k = 0; k < num_tiles; k++)
      {
        for(int ii = 0; ii < TS; ii++)
        {
          int a_m_idx = (i * TS + ii) * AW + k * TS;
          for(int jj = 0; jj < TS; jj++)
          {
            int rlt_idx = (i * TS + ii) * BW + j * TS + jj;
            int b_m_idx = k * TS * BW  + j * TS + jj;

            for(int idx = 0; idx < TS; idx++)
              result[rlt_idx] += a[a_m_idx + idx] * b[b_m_idx+ idx * BW];
          }
        }
      }
    }
  }
  return;
}

int main(int argc, char **argv)
{
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

  // create opencl program for .cl kernel source
  program = CreateProgram(context, device, "kernel_tvm.cl");
  if(program == NULL)
  {
    Cleanup(context, commandQueue, program, kernel, memObjects);
    return 1;
  }

  // create opencl kernel
  kernel = clCreateKernel(program, "rcl__kernel0", NULL);
  if(kernel == NULL)
  {
    cerr << "Failed to create kernel!" << endl;
    Cleanup(context, commandQueue, program, kernel, memObjects);
    return 1;
  }

  float *a, *b;
  float *result;
  int a_height, a_width, b_width;

  a_height = AH;
  a_width = AW;
  b_width = BW;

  a = (float *)malloc(sizeof(float) * SIZE_A);
  b = (float *)malloc(sizeof(float) * SIZE_B);
  result = (float *)malloc(sizeof(float) * SIZE_C);

  for(int idx = 0; idx < SIZE_A; idx++)
    a[idx] = rand() % 3;
  for(int idx = 0; idx < SIZE_B; idx++)
    b[idx] = rand() % 3;
  for(int idx = 0; idx < SIZE_C; idx++)
    result[idx] = 0.0f;

  double s_time, e_time;
  s_time = timer();
  mat_mul(a, b, result);
  e_time = timer();
  cout << "Time used by mat_mul: " << e_time - s_time << endl;

  if(!compare_result(a, b, result))
    cout << "The result of mat_mul is wrong!" << endl;
  else
    cout << "The result of mat_mul is correct!" << endl;

  for(int idx = 0; idx < SIZE_C; idx++)
    result[idx] = 0.0f;

  s_time = timer();
  tiled_mat_mul(a, b, result);
  e_time = timer();
  cout << "Time used by tiled_mat_mul: " << e_time - s_time << endl;

  if(!compare_result(a, b, result))
    cout << "The result of tiled_mat_mul is wrong!" << endl;
  else
    cout << "The result of tiled_mat_mul is correct!" << endl;

  for(int idx = 0; idx < SIZE_C; idx++)
    result[idx] = 0.0f;

  memObjects[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                 sizeof(float) * SIZE_A, a, NULL);
  memObjects[1] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                 sizeof(float) * SIZE_B, b, NULL);
  memObjects[2] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                                 sizeof(float) * SIZE_C, result, NULL);

  if(memObjects[0] == NULL || memObjects[1] == NULL ||
     memObjects[2] == NULL)
  {
    cerr << "Error creating memory objects!" << endl;
    Cleanup(context, commandQueue, program, kernel, memObjects);
    return 1;
  }

  size_t globalWorkSize[1] = {AH};
  size_t localWorkSize[1] = {TS};
 
  errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &memObjects[0]);
  errNum = clSetKernelArg(kernel, 1, sizeof(cl_mem), &memObjects[1]);
  errNum = clSetKernelArg(kernel, 2, sizeof(cl_mem), &memObjects[2]);

  if(errNum != CL_SUCCESS)
  {
    cerr << "Error setting kernel arguments!"  << endl;
    Cleanup(context, commandQueue, program, kernel, memObjects);
    return 1;
  }

  s_time = timer();
  errNum = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, 
                                  globalWorkSize, localWorkSize,
                                  0, NULL, &event);
  
  e_time = timer();
  cout << "Time used by opencl: " << e_time - s_time << endl;
  if(errNum != CL_SUCCESS)
  {
    cerr << "clEnqueueNDRangeKernel!" << endl;
    Cleanup(context, commandQueue, program, kernel, memObjects);
    return 1;
  }

  errNum = clWaitForEvents(1, &event);
  if(errNum != CL_SUCCESS)
  {
    cerr << "clWaitForEvents" << endl;
    Cleanup(context, commandQueue, program, kernel, memObjects);
    return 1;
  }


  errNum = clEnqueueReadBuffer(commandQueue, memObjects[2],
                               CL_TRUE, 0, SIZE_C * sizeof(float),
                               result, 0, NULL, NULL);
  
  if(errNum != CL_SUCCESS)
  {
    cerr << "clEnqueueReadBuffer"  << endl;
    Cleanup(context, commandQueue, program, kernel, memObjects);
    return 1;
  }
  cout << "Executed program successfully." << endl;

  if(!compare_result(a, b, result))
    cout << "The result of opencl is wrong!" << endl;
  else
    cout << "The result of opencl is correct!" << endl;

  Cleanup(context, commandQueue, program, kernel, memObjects);

  free(a);
  free(b);
  free(result);

  return 0;
}

