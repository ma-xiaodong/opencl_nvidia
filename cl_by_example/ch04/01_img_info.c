#include <stdio.h>
#include <CL/cl.h>

int main(void)
{
  cl_platform_id *platforms = NULL;
  cl_uint num_platforms;
  cl_int cl_status = clGetPlatformIDs(0, NULL, &num_platforms);
  platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * num_platforms);
  cl_status = clGetPlatformIDs(num_platforms, platforms, NULL);

  cl_device_id *device_list = NULL;
  cl_uint num_devices;

  cl_status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
  device_list = (cl_device_id *)malloc(sizeof(cl_device_id) * num_devices);
  cl_status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, num_devices,
                             device_list, NULL);
  // query image support
  cl_bool image_support = CL_FALSE;
  cl_status = clGetDeviceInfo(device_list[0], CL_DEVICE_IMAGE_SUPPORT, 
                              sizeof(cl_bool), &image_support, NULL);
  if(image_support)
    printf("Support image!\n");
  else
  {
    printf("Do not support image!\n");
    return 0;
  }

  cl_context context;
  cl_uint num_img_formats;

  context = clCreateContext(NULL, num_devices, device_list, NULL, NULL, &cl_status);
  cl_status = clGetSupportedImageFormats(context, CL_MEM_READ_ONLY,
                                         CL_MEM_OBJECT_IMAGE2D, 0, NULL, &num_img_formats);
  printf("number of image formats: %d.\n", num_img_formats);

  cl_status = clReleaseContext(context);

  free(platforms);
  free(device_list);
  return 0;
}

