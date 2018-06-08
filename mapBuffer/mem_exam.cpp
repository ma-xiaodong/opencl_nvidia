#include <iostream>
#include "gemm.hpp"

int main(int argc, char **argv)
{
  cl_context context = 0;
  cl_command_queue cmdque = 0;
  cl_program program = 0;
  cl_mem memobj;
  cl_event k_event;
  cl_device_id device = 0;
  cl_kernel kernel = 0;
  cl_int errnum;

  context = CreateContext();

  cmdque = CreateCommandQueue(context, &device);
  if(cmdque == NULL)
  {
    std::cout << "CreateCommandQueue failed!" << std::endl;
    Cleanup(context, cmdque, program, kernel, &memobj, &memobj, &memobj);
    return 0;
  }

  program = CreateProgram(context, device, "mem_exam.cl");
  if(program == NULL)
  {
    std::cout << "CreateProgram failed!" << std::endl;
    Cleanup(context, cmdque, program, kernel, &memobj, &memobj, &memobj);
    return 0; 
  }

  kernel = clCreateKernel(program, "float_add", NULL);
  if(kernel == NULL)
  {
    std::cout << "clCreateKernel failed!" << std::endl;
    Cleanup(context, cmdque, program, kernel, &memobj, &memobj, &memobj);
    return 0; 
  }
  memobj = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE, 
                          sizeof(float) * 20, NULL, NULL);
  errnum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &memobj);
  if(errnum != CL_SUCCESS)
  {
    std::cout << "clSetKernelArg failed!"  << std::endl;
    Cleanup(context, cmdque, program, kernel, &memobj, &memobj, &memobj);
    return 0;
  }

  size_t global_size[1] = {20};
  errnum = clEnqueueNDRangeKernel(cmdque, kernel, 1, NULL, 
                                  global_size, NULL, 0, NULL, &k_event);
  if(errnum != CL_SUCCESS)
  { 
    std::cout << "clEnqueueNDRangeKernel failed!"  << std::endl;
    Cleanup(context, cmdque, program, kernel, &memobj, &memobj, &memobj);
    return 0;
  }

  errnum = clWaitForEvents(1, &k_event);
  if(errnum != CL_SUCCESS)
  { 
    std::cout << "clWaitForEvents 0 failed!"  << std::endl;
    Cleanup(context, cmdque, program, kernel, &memobj, &memobj, &memobj);
    return 0;
  }
 
  float *host_ptr = (float *)clEnqueueMapBuffer(cmdque, memobj, CL_FALSE, CL_MAP_READ,
                                       0, sizeof(float) * 20, 0, NULL, &k_event, NULL);
 
  errnum = clWaitForEvents(1, &k_event);
  if(errnum != CL_SUCCESS)
  { 
    std::cout << "clWaitForEvents 1 failed!"  << std::endl;
    Cleanup(context, cmdque, program, kernel, &memobj, &memobj, &memobj);
    return 0;
  }

  std::cout << host_ptr << ": " << memobj << std::endl;
  for(int idx = 0; idx < 20; idx++)
  {
    std::cout << host_ptr[idx] << " ";
  }
  std::cout << std::endl;
}
