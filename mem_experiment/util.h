#include <CL/cl.h>

#define AH      (1024)
#define AW      (1024)
#define BH      AW
#define BW      (1024)
#define BLK	(8)

void get_perf_info(const char *msg, cl_event *event, bool flops);
void setup_cl();
void clean_cl();
cl_program CreateProgram(cl_context context, cl_device_id device, 
                         const char *fileName, const char *build_opt);

