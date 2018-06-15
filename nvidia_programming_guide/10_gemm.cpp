#include <stdio.h>
#include <CL/cl.h>
#include <string.h>
#include <stdlib.h>
#include <iostream>
#include "12_util.h"

using namespace std;

extern cl_platform_id __platform;
extern cl_device_id __device;
extern cl_context __context;
extern cl_command_queue __command_queue;

float *a, *b, *c, *std_c;

int compare_result(float *result, float *std_result, int size)
{
    // only compare a random point
    int flag = 1;
    for(int ii = 0; ii < size; ii++)
    {
        if((result[ii] - std_result[ii]) < -1e-4 || (result[ii] - std_result[ii]) > 1e-4)
        {
            flag = 0;
            cout << "Result error: [" << ii << "], ";
            cout << result[ii] << " : " << std_result[ii] << endl;
        }
    }
    return flag;
}

void naive()
{
    cl_int cl_status;
    cl_program program;
    cl_kernel kernel;
    cl_mem mem_a, mem_b, mem_c;
    cl_event event;
    size_t global_size[2], local_size[2];
    char build_opt[64] = "\0";

    sprintf(build_opt, "%s%d %s%d", "-DAW=", AW, "-DBW=", BW);
    strcat(build_opt, "\0");
    
    program = CreateProgram(__context, __device, "01_naive.cl", build_opt);
    kernel = clCreateKernel(program, "naive", &cl_status);
    if(cl_status != CL_SUCCESS)
    {
	cout << "Error: clCreateKernel!" << endl;
	clReleaseProgram(program);
	return;
    }

    mem_a = clCreateBuffer(__context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 
                           sizeof(float) * AH * AW, a, &cl_status);
    mem_b = clCreateBuffer(__context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 
                           sizeof(float) * BH * BW, b, &cl_status);
    mem_c = clCreateBuffer(__context, CL_MEM_WRITE_ONLY, 
                           sizeof(float) * AH * BW, NULL, &cl_status);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&mem_a);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&mem_b);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&mem_c);

    global_size[0] = BW; global_size[1] = AH;
    local_size[0] = GRP_SZ; local_size[1] = GRP_SZ;

    cl_status = clEnqueueNDRangeKernel(__command_queue, kernel, 2, NULL, global_size, 
                                       local_size, 0, NULL, &event);

    clWaitForEvents(1, &event);
    get_perf_info("Gemm naive run", &event, true);

    cl_status = clEnqueueReadBuffer(__command_queue, mem_c, CL_FALSE, 0, 
                                    sizeof(float) * AH * BW, std_c, 0, NULL, &event);
    clWaitForEvents(1, &event);
    get_perf_info("Gemm naive read buffer", &event, false);

    clReleaseEvent(event);
    clReleaseMemObject(mem_a);
    clReleaseMemObject(mem_b);
    clReleaseMemObject(mem_c);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
}

void local()
{
    cl_int cl_status;
    cl_program program;
    cl_kernel kernel;
    cl_mem mem_a, mem_b, mem_c;
    cl_event event;
    size_t global_size[2], local_size[2];
    char build_opt[64] = "\0";

    sprintf(build_opt, "%s%d %s%d %s%d", "-DAW=", AW, "-DBW=", BW, "-DLCL_SZ=", LCL_SZ);
    strcat(build_opt, "\0");

    program = CreateProgram(__context, __device, "02_local.cl", build_opt);
    kernel = clCreateKernel(program, "local_opt", &cl_status);
    if(cl_status != CL_SUCCESS)
    {
	cout << "Error: clCreateKernel!" << endl;
	clReleaseProgram(program);
	return;
    }

    mem_a = clCreateBuffer(__context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 
                           sizeof(float) * AH * AW, a, &cl_status);
    mem_b = clCreateBuffer(__context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 
                           sizeof(float) * BH * BW, b, &cl_status);
    mem_c = clCreateBuffer(__context, CL_MEM_WRITE_ONLY, 
                           sizeof(float) * AH * BW, NULL, &cl_status);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&mem_a);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&mem_b);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&mem_c);

    global_size[0] = BW; global_size[1] = AH;
    local_size[0] = LCL_SZ; local_size[1] = LCL_SZ;

    cl_status = clEnqueueNDRangeKernel(__command_queue, kernel, 2, NULL, global_size, 
                                       local_size, 0, NULL, &event);

    clWaitForEvents(1, &event);
    get_perf_info("Gemm local run", &event, true);

    cl_status = clEnqueueReadBuffer(__command_queue, mem_c, CL_FALSE, 0, 
                                    sizeof(float) * AH * BW, c, 0, NULL, &event);
    clWaitForEvents(1, &event);
    get_perf_info("Gemm local read buffer", &event, false);
    if(!compare_result(c, std_c, AH * BW))
	cout << "Result wrong!" << endl;
    else
	cout << "Result correct!" << endl;

    clReleaseEvent(event);
    clReleaseMemObject(mem_a);
    clReleaseMemObject(mem_b);
    clReleaseMemObject(mem_c);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
}

void lclwpt()
{
    cl_int cl_status;
    cl_program program;
    cl_kernel kernel;
    cl_mem mem_a, mem_b, mem_c;
    cl_event event;
    size_t global_size[2], local_size[2];
    char build_opt[64] = "\0";

    sprintf(build_opt, "%s%d %s%d %s%d %s%d", "-DAW=", AW, "-DBW=", BW, "-DLCL_SZ=", LCL_SZ,
            "-DWPT=", WPT);
    strcat(build_opt, "\0");

    program = CreateProgram(__context, __device, "03_lclwpt.cl", build_opt);
    kernel = clCreateKernel(program, "lclwpt_opt", &cl_status);
    if(cl_status != CL_SUCCESS)
    {
	cout << "Error: clCreateKernel!" << endl;
	clReleaseProgram(program);
	return;
    }

    mem_a = clCreateBuffer(__context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 
                           sizeof(float) * AH * AW, a, &cl_status);
    mem_b = clCreateBuffer(__context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 
                           sizeof(float) * BH * BW, b, &cl_status);
    mem_c = clCreateBuffer(__context, CL_MEM_WRITE_ONLY, 
                           sizeof(float) * AH * BW, NULL, &cl_status);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&mem_a);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&mem_b);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&mem_c);

    global_size[0] = BW / WPT; global_size[1] = AH;
    local_size[0] = LCL_SZ / WPT; local_size[1] = LCL_SZ;

    cl_status = clEnqueueNDRangeKernel(__command_queue, kernel, 2, NULL, global_size, 
                                       local_size, 0, NULL, &event);

    clWaitForEvents(1, &event);
    get_perf_info("Gemm lclwpt run", &event, true);

    cl_status = clEnqueueReadBuffer(__command_queue, mem_c, CL_FALSE, 0, 
                                    sizeof(float) * AH * BW, c, 0, NULL, &event);
    clWaitForEvents(1, &event);
    get_perf_info("Gemm lclwpt read buffer", &event, false);
    if(!compare_result(c, std_c, AH * BW))
	cout << "Result wrong!" << endl;
    else
	cout << "Result correct!" << endl;

    clReleaseEvent(event);
    clReleaseMemObject(mem_a);
    clReleaseMemObject(mem_b);
    clReleaseMemObject(mem_c);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
}

// The vector size is set 4 by default.
void lclvec()
{
    cl_int cl_status;
    cl_program program;
    cl_kernel kernel;
    cl_mem mem_a, mem_b, mem_c;
    cl_event event;
    size_t global_size[2], local_size[2];
    char build_opt[64] = "\0";

    sprintf(build_opt, "%s%d %s%d %s%d", "-DAW=", AW, "-DBW=", BW, "-DLCL_SZ=", LCL_SZ);
    strcat(build_opt, "\0");

    program = CreateProgram(__context, __device, "04_lclvec.cl", build_opt);
    kernel = clCreateKernel(program, "lclvec_opt", &cl_status);
    if(cl_status != CL_SUCCESS)
    {
	cout << "Error: clCreateKernel!" << endl;
	clReleaseProgram(program);
	return;
    }

    mem_a = clCreateBuffer(__context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 
                           sizeof(float) * AH * AW, a, &cl_status);
    mem_b = clCreateBuffer(__context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 
                           sizeof(float) * BH * BW, b, &cl_status);
    mem_c = clCreateBuffer(__context, CL_MEM_WRITE_ONLY, 
                           sizeof(float) * AH * BW, NULL, &cl_status);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&mem_a);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&mem_b);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&mem_c);

    global_size[0] = BW / 4; global_size[1] = AH;
    local_size[0] = LCL_SZ / 4; local_size[1] = LCL_SZ;

    cl_status = clEnqueueNDRangeKernel(__command_queue, kernel, 2, NULL, global_size, 
                                       local_size, 0, NULL, &event);

    clWaitForEvents(1, &event);
    get_perf_info("Gemm lclvec run", &event, true);

    cl_status = clEnqueueReadBuffer(__command_queue, mem_c, CL_FALSE, 0, 
                                    sizeof(float) * AH * BW, c, 0, NULL, &event);
    clWaitForEvents(1, &event);
    get_perf_info("Gemm lclvec read buffer", &event, false);
    if(!compare_result(c, std_c, AH * BW))
	cout << "Result wrong!" << endl;
    else
	cout << "Result correct!" << endl;

    clReleaseEvent(event);
    clReleaseMemObject(mem_a);
    clReleaseMemObject(mem_b);
    clReleaseMemObject(mem_c);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
}

void lclvec_trans()
{
    cl_int cl_status;
    cl_program program;
    cl_kernel kernel[2];
    cl_mem mem_a, mem_b, mem_c, mem_trans_b;
    cl_event event[3];
    size_t global_size[2], local_size[2];

    // Transpose B firstly.
    mem_a = clCreateBuffer(__context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 
                           sizeof(float) * AH * AW, a, &cl_status);
    mem_b = clCreateBuffer(__context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 
                           sizeof(float) * BH * BW, b, &cl_status);
    mem_c = clCreateBuffer(__context, CL_MEM_WRITE_ONLY, 
                           sizeof(float) * AH * BW, NULL, &cl_status);
    mem_trans_b = clCreateBuffer(__context, CL_MEM_READ_WRITE, 
                                 sizeof(float) * AH * BW, NULL, &cl_status);

    char build_opt[64] = "\0";
    sprintf(build_opt, "%s%d %s%d %s%d", "-DAW=", AW, "-DBW=", BW, "-DLCL_SZ=", LCL_SZ);
    strcat(build_opt, "\0");

    program = CreateProgram(__context, __device, "05_lclvec_trans.cl", build_opt);

    kernel[0] = clCreateKernel(program, "transpose", &cl_status);
    if(cl_status != CL_SUCCESS)
    {
	cout << "Error: clCreateKernel transpose!" << endl;
	clReleaseProgram(program);
	return;
    }
    clSetKernelArg(kernel[0], 0, sizeof(cl_mem), (void *)&mem_b);
    clSetKernelArg(kernel[0], 1, sizeof(cl_mem), (void *)&mem_trans_b);
    
    global_size[0] = BW; global_size[1] = AW;
    local_size[0] = LCL_SZ; local_size[1] = LCL_SZ;

    cl_status = clEnqueueNDRangeKernel(__command_queue, kernel[0], 2, NULL, global_size, 
                                       local_size, 0, NULL, &(event[0]));

    // Matrix multiplication.
    kernel[1] = clCreateKernel(program, "lclvec_trans_opt", &cl_status);
    if(cl_status != CL_SUCCESS)
    {
	cout << "Error: clCreateKernel lcl_vec_trans_opt!" << endl;
	clReleaseProgram(program);
	return;
    }
    clSetKernelArg(kernel[1], 0, sizeof(cl_mem), (void *)&mem_a);
    clSetKernelArg(kernel[1], 1, sizeof(cl_mem), (void *)&mem_trans_b);
    clSetKernelArg(kernel[1], 2, sizeof(cl_mem), (void *)&mem_c);

    global_size[0] = BW / 4; global_size[1] = AH;
    local_size[0] = LCL_SZ / 4; local_size[1] = LCL_SZ;

    cl_status = clEnqueueNDRangeKernel(__command_queue, kernel[1], 2, NULL, global_size, 
                                       local_size, 1, &(event[0]), &(event[1]));

    cl_status = clEnqueueReadBuffer(__command_queue, mem_c, CL_FALSE, 0, 
                                    sizeof(float) * AH * BW, c, 1, &(event[1]), &(event[2]));
    clWaitForEvents(1, &(event[2]));

    get_perf_info("Gemm lclvec_trans trnaspose", &(event[0]), false);
    get_perf_info("Gemm lclvec_trans run", &(event[1]), true);
    get_perf_info("Gemm lclvec read buffer", &(event[2]), false);

/*
    if(!compare_result(c, std_c, AH * BW))
	cout << "Result wrong!" << endl;
    else
	cout << "Result correct!" << endl;
*/

    clReleaseEvent(event[0]);
    clReleaseEvent(event[1]);
    clReleaseEvent(event[2]);
    clReleaseMemObject(mem_a);
    clReleaseMemObject(mem_b);
    clReleaseMemObject(mem_c);
    clReleaseMemObject(mem_trans_b);
    clReleaseKernel(kernel[0]);
    clReleaseKernel(kernel[1]);
    clReleaseProgram(program);
}

int main(int argc, char **argv)
{
    setup_cl();
    a = (float *)malloc(sizeof(float) * AH * AW);
    b = (float *)malloc(sizeof(float) * BH * BW);
    c = (float *)malloc(sizeof(float) * AH * BW);
    std_c = (float *)malloc(sizeof(float) * AH * BW);
     
    for(int idx = 0; idx < AH * AW; idx++)
	a[idx] = rand() % 3 / 5.1;
    for(int idx = 0; idx < BH * BW; idx++)
	b[idx] = rand() % 3 / 7.1;
    for(int idx = 0; idx < AH * BW; idx++)
    {
        c[idx] = 0.0f;
        std_c[idx] = 0.0f;
    }
    // Naive gpu gemm
    naive();

    // Optimization using local mem.
    for(int idx = 0; idx < AH * BW; idx++)
        c[idx] = 0.0f;
    local();

    // Optimization using local && wpt.
    for(int idx = 0; idx < AH * BW; idx++)
        c[idx] = 0.0f;
    lclwpt();

    // Optimization using local && vectorization.
    for(int idx = 0; idx < AH * BW; idx++)
        c[idx] = 0.0f;
    lclvec();

    // Optimization using local && vectorization && transpose.
    for(int idx = 0; idx < AH * BW; idx++)
        c[idx] = 0.0f;
    lclvec_trans();

    // Release resource
    free(a); free(b); free(c); free(std_c);
    clean_cl();
}

