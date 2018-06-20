#include <stdio.h>
#include <CL/cl.h>
#include <string.h>
#include <stdlib.h>
#include <iostream>
#include <sys/time.h>
#include "12_util.h"

using namespace std;

extern cl_platform_id __platform;
extern cl_device_id __device;
extern cl_context __context;
extern cl_command_queue __command_queue;

float *a, *b, *c, *std_c;
double s_time, e_time;

double timer(void)
{
    struct timeval Tvalue;
    struct timezone dummy;
    
    gettimeofday(&Tvalue, &dummy);
    double etime = (double)Tvalue.tv_sec + 1.0e-6*((double)Tvalue.tv_usec);
    return etime * 1.0e3;
}

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
    cl_event event[2];
    size_t global_size[2], local_size[2];
    char build_opt[64] = "\0";

    cout << "\n++++ 01 naive run ++++" << endl;
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

    s_time = timer();
    cl_status = clEnqueueNDRangeKernel(__command_queue, kernel, 2, NULL, global_size, 
                                       local_size, 0, NULL, &(event[0]));
    cl_status = clEnqueueReadBuffer(__command_queue, mem_c, CL_FALSE, 0, 
                                    sizeof(float) * AH * BW, std_c, 1, &(event[0]), &(event[1]));
    clWaitForEvents(1, &(event[1]));
    e_time = timer();
    cout << "Clock time: " << e_time - s_time << "ms" << endl;
    get_perf_info("Gemm naive run", &(event[0]), true);
    get_perf_info("Gemm naive read buffer", &(event[1]), false);

    clReleaseEvent(event[0]);
    clReleaseEvent(event[1]);
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
    cl_event event[2];
    size_t global_size[2], local_size[2];
    char build_opt[64] = "\0";

    cout << "\n++++ 02 local mem optimization ++++" << endl;
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

    s_time = timer();
    cl_status = clEnqueueNDRangeKernel(__command_queue, kernel, 2, NULL, global_size, 
                                       local_size, 0, NULL, &(event[0]));
    cl_status = clEnqueueReadBuffer(__command_queue, mem_c, CL_FALSE, 0, 
                                    sizeof(float) * AH * BW, c, 1, &(event[0]), &(event[1]));
    clWaitForEvents(1, &(event[1]));
    e_time = timer();
 
    cout << "Clock time: " << e_time - s_time << "ms" << endl;
    get_perf_info("Gemm local run", &(event[0]), true);
    get_perf_info("Gemm local read buffer", &(event[1]), false);

    if(!compare_result(c, std_c, AH * BW))
	cout << "Result wrong!" << endl;
    else
	cout << "Result correct!" << endl;

    clReleaseEvent(event[0]);
    clReleaseEvent(event[1]);
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
    cl_event event[2];
    size_t global_size[2], local_size[2];
    char build_opt[64] = "\0";

    cout << "\n++++ 03 local mem && work per tile optimization ++++" << endl;
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

    s_time = timer();
    cl_status = clEnqueueNDRangeKernel(__command_queue, kernel, 2, NULL, global_size, 
                                       local_size, 0, NULL, &(event[0]));
    cl_status = clEnqueueReadBuffer(__command_queue, mem_c, CL_FALSE, 0, 
                                    sizeof(float) * AH * BW, c, 1, &(event[0]), &(event[1]));
    clWaitForEvents(1, &(event[1]));
    e_time = timer();

    cout << "Clock time: " << e_time - s_time << "ms" << endl;
    get_perf_info("Gemm lclwpt run", &(event[0]), true);
    get_perf_info("Gemm lclwpt read buffer", &(event[1]), false);

    if(!compare_result(c, std_c, AH * BW))
	cout << "Result wrong!" << endl;
    else
	cout << "Result correct!" << endl;

    clReleaseEvent(event[0]);
    clReleaseEvent(event[1]);
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
    cl_event event[2];
    size_t global_size[2], local_size[2];
    char build_opt[64] = "\0";

    cout << "\n++++ 04 local mem && vectorization optimization ++++" << endl;
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

    s_time = timer();
    cl_status = clEnqueueNDRangeKernel(__command_queue, kernel, 2, NULL, global_size, 
                                       local_size, 0, NULL, &(event[0]));
    cl_status = clEnqueueReadBuffer(__command_queue, mem_c, CL_FALSE, 0, 
                                    sizeof(float) * AH * BW, c, 1, &(event[0]), &(event[1]));
    clWaitForEvents(1, &(event[1]));
    e_time = timer();

    cout << "Clock time: " << e_time - s_time << "ms" << endl;
    get_perf_info("Gemm lclvec run", &(event[0]), true);
    get_perf_info("Gemm lclvec read buffer", &(event[1]), false);

    if(!compare_result(c, std_c, AH * BW))
	cout << "Result wrong!" << endl;
    else
	cout << "Result correct!" << endl;

    clReleaseEvent(event[0]);
    clReleaseEvent(event[1]);
    clReleaseMemObject(mem_a);
    clReleaseMemObject(mem_b);
    clReleaseMemObject(mem_c);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
}

void lclvec_rt()
{
    cl_int cl_status;
    cl_program program;
    cl_kernel kernel;
    cl_mem mem_a, mem_b, mem_c;
    cl_event event[2];
    size_t global_size[2], local_size[2];
    char build_opt[64] = "\0";

    cout << "\n++++ 05 local mem && vectorization && register tiling optimization ++++" << endl;
    sprintf(build_opt, "%s%d %s%d %s%d", "-DAW=", AW, "-DBW=", BW, "-DLCL_SZ=", LCL_SZ);
    strcat(build_opt, "\0");

    program = CreateProgram(__context, __device, "05_lclvec_rt.cl", build_opt);
#ifdef VEC4X4
    kernel = clCreateKernel(program, "lclvec4x4_rt_opt", &cl_status);
#elif defined VEC4X8
    kernel = clCreateKernel(program, "lclvec4x8_rt_opt", &cl_status);
#endif

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

#ifdef VEC4X4
    // Each workitem compute a 4 * 4 tile of the final matrix.
    global_size[0] = BW / 4; global_size[1] = AH / 4;
    local_size[0] = LCL_SZ / 4; local_size[1] = LCL_SZ / 4;
#elif defined VEC4X8
    // Each workitem compute a 4 * 4 tile of the final matrix.
    global_size[0] = BW / 8; global_size[1] = AH / 4;
    local_size[0] = LCL_SZ / 8; local_size[1] = LCL_SZ / 4;
#endif

    s_time = timer();
    cl_status = clEnqueueNDRangeKernel(__command_queue, kernel, 2, NULL, global_size, 
                                       local_size, 0, NULL, &(event[0]));
    
    cl_status = clEnqueueReadBuffer(__command_queue, mem_c, CL_FALSE, 0, 
                                    sizeof(float) * AH * BW, c, 1, &(event[0]), &(event[1]));
    clWaitForEvents(1, &event[1]);
    e_time = timer();

    cout << "Clock time: " << e_time - s_time << "ms" << endl;
    get_perf_info("Gemm lclvec_rt run", &(event[0]), true);
    get_perf_info("Gemm lclvec_rt read buffer", &(event[1]), false);

    if(!compare_result(c, std_c, AH * BW))
	cout << "Result wrong!" << endl;
    else
	cout << "Result correct!" << endl;

    clReleaseEvent(event[0]);
    clReleaseEvent(event[1]);
    clReleaseMemObject(mem_a);
    clReleaseMemObject(mem_b);
    clReleaseMemObject(mem_c);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
}

void lclvec_rtrec()
{
    cl_int cl_status;
    cl_program program;
    cl_kernel kernel;
    cl_mem mem_a, mem_b, mem_c;
    cl_event event[2];
    size_t global_size[2], local_size[2];
    char build_opt[64] = "\0";

    cout << "\n++++ 06 rectangular local mem && vectorization && register tiling optimization ++++" << endl;
    sprintf(build_opt, "%s%d %s%d %s%d %s%d", "-DAW=", AW, "-DBW=", BW, "-DLCL_MN=", LCL_MN, 
            "-DMN_TIMES=", MN_TIMES);
    strcat(build_opt, "\0");

    program = CreateProgram(__context, __device, "06_lclvec_rtrec.cl", build_opt);
    kernel = clCreateKernel(program, "lclvec_rtrec_opt", &cl_status);
    if(cl_status != CL_SUCCESS)
    {
	cout << "Error: clCreateKernel lclvec_rtrec_opt!" << endl;
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

    // Each workitem compute a 4 * 4 tile of the final matrix.
    global_size[0] = BW / 4; global_size[1] = AH / 4;
    local_size[0] = LCL_MN / 4; local_size[1] = LCL_MN / 4;

    s_time = timer();
    cl_status = clEnqueueNDRangeKernel(__command_queue, kernel, 2, NULL, global_size, 
                                       local_size, 0, NULL, &(event[0]));
    
    cl_status = clEnqueueReadBuffer(__command_queue, mem_c, CL_FALSE, 0, 
                                    sizeof(float) * AH * BW, c, 1, &(event[0]), &(event[1]));
    clWaitForEvents(1, &event[1]);
    e_time = timer();

    cout << "Clock time: " << e_time - s_time << "ms" << endl;
    get_perf_info("Gemm lclvec_rtrec run", &(event[0]), true);
    get_perf_info("Gemm lclvec_rtrec read buffer", &(event[1]), false);

    if(!compare_result(c, std_c, AH * BW))
	cout << "Result wrong!" << endl;
    else
	cout << "Result correct!" << endl;

    clReleaseEvent(event[0]);
    clReleaseEvent(event[1]);
    clReleaseMemObject(mem_a);
    clReleaseMemObject(mem_b);
    clReleaseMemObject(mem_c);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
}

/****************
Parameters:
  1. tile size of dimension M and N (they have the same tile size)
  2. tile size of dimension K
****************/

void tuning(int tile_mn, int tile_k)
{
    cl_int cl_status;
    cl_program program;
    cl_kernel kernel;
    cl_mem mem_a, mem_b, mem_c;
    cl_event event[2];
    size_t global_size[2], local_size[2];
    char build_opt[64] = "\0";

    cout << "\n++++ 07 gemm tuning on nvidia ++++" << endl;
    sprintf(build_opt, "%s%d %s%d %s%d %s%d", "-DAW=", AW, "-DBW=", BW, "-DTILE_MN=", tile_mn, 
            "-DTILE_K=", tile_k);
    strcat(build_opt, "\0");

    program = CreateProgram(__context, __device, "07_tuning.cl", build_opt);
    kernel = clCreateKernel(program, "tuning4x4_opt", &cl_status);
    if(cl_status != CL_SUCCESS)
    {
	cout << "Error: clCreateKernel tuning4x4_opt!" << endl;
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

    global_size[0] = BW / 4; global_size[1] = AH / 4;
    local_size[0] = tile_mn / 4; local_size[1] = tile_mn / 4;

    s_time = timer();
    cl_status = clEnqueueNDRangeKernel(__command_queue, kernel, 2, NULL, global_size, 
                                       local_size, 0, NULL, &(event[0]));
    
    cl_status = clEnqueueReadBuffer(__command_queue, mem_c, CL_FALSE, 0, 
                                    sizeof(float) * AH * BW, c, 1, &(event[0]), &(event[1]));
    clWaitForEvents(1, &event[1]);
    e_time = timer();

    cout << "Clock time: " << e_time - s_time << "ms" << endl;
    get_perf_info("Gemm tuning run", &(event[0]), true);
    get_perf_info("Gemm tuning read buffer", &(event[1]), false);

    if(!compare_result(c, std_c, AH * BW))
	cout << "Result wrong!" << endl;
    else
	cout << "Result correct!" << endl;

    clReleaseEvent(event[0]);
    clReleaseEvent(event[1]);
    clReleaseMemObject(mem_a);
    clReleaseMemObject(mem_b);
    clReleaseMemObject(mem_c);
    clReleaseKernel(kernel);
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
    
    // Optimization using local && vectorization && register tiling.
    for(int idx = 0; idx < AH * BW; idx++)
        c[idx] = 0.0f;
    lclvec_rt();

    // Optimization using local && vectorization && register tiling.
    for(int idx = 0; idx < AH * BW; idx++)
        c[idx] = 0.0f;
    lclvec_rtrec();

    // Optimization using local && vectorization && register tiling.
    for(int idx = 0; idx < AH * BW; idx++)
        c[idx] = 0.0f;
    tuning(32, 16);

    // Release resource
    free(a); free(b); free(c); free(std_c);
    clean_cl();
}

