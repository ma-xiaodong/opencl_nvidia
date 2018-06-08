#include <CL/cl.h>

// gpu utils functions
void handle_error(cl_int ret, bool flag, const char *msg);
cl_context CreateContext();
cl_command_queue CreateCommandQueue(cl_context context, cl_device_id *device);
cl_program CreateProgram(cl_context context, cl_device_id device, const char *fileName);
void Cleanup(cl_context context, cl_command_queue cmdQueue, cl_program program, cl_kernel kernel, cl_mem *memObj0, cl_mem *memObj1, cl_mem *memObj2);

