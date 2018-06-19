#include <CL/cl.h>

#define AH		(2048)
#define AW		(2048)
#define BH		AW
#define BW		(2048)

#define GRP_SZ          16
#define LCL_SZ          32
#define WPT             8

// rectangular tiling
#define LCL_MN		32
#define MN_TIMES	2

void get_perf_info(const char *msg, cl_event *event, bool flops);
void setup_cl();
void clean_cl();
cl_program CreateProgram(cl_context context, cl_device_id device, 
                         const char *fileName, const char *build_opt);

