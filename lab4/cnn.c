#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>
#include "cnn.h"
#include <string.h>

#include <CL/cl.h>
#include "kernel_cl.h"

#define BLOCK_SIZE 32

inline void checkErr(cl_int err, const char * name) {
	if (err != CL_SUCCESS) {
		fprintf(stderr, "ERROR: %s (%d)\n", name, err);
		exit(EXIT_FAILURE);
	}
}

// Sequential CNN implementation
void conv(float Cout[NUM][OUTIMROW][OUTIMROW], float Cin[NUM][INIMROW][INIMROW],
          float weight[NUM][NUM][KERNEL][KERNEL], float bias[NUM])
{
	static float C[NUM][IMROW][IMROW];

	for(int i = 0; i < NUM; i++) {
		for(int h = 0; h < IMROW; h++) {
			for(int w = 0; w < IMROW; w++)
				C[i][h][w] = bias[i];
		}
	}

// Convolution
	for(int i = 0; i < NUM; i++) {
		for(int j = 0; j < NUM; j++) {
			for(int h = 0; h < IMROW; h++) {
				for(int w = 0; w < IMROW; w++) {
					for(int p = 0; p < KERNEL; p++) {
						for(int q = 0; q < KERNEL; q++)
							C[i][h][w] += weight[i][j][p][q] * Cin[j][h + p][w + q];
					}
				}
			}
		}
	}

// ReLU
	for (int i = 0; i < NUM; i++) {
		for (int h = 0; h < IMROW; h++) {
			for (int w = 0; w < IMROW; w++) {
				C[i][h][w] = fmax(0, C[i][h][w]);
			}
		}
	}

// Max pooling
	for (int i = 0; i < NUM; i++) {
		for (int h = 0; h < OUTIMROW; h++) {
			for (int w = 0; w < OUTIMROW; w++) {
				float local_max = C[i][2 * h][2 * w];
				local_max = fmax(local_max, C[i][2 * h + 1][2 * w]);
				local_max = fmax(local_max, C[i][2 * h + 1][2 * w + 1]);
				local_max = fmax(local_max, C[i][2 * h][2 * w + 1]);
				Cout[i][h][w] = local_max;
			}
		}
	}
}

void opencl_gpu(
   float Cout[NUM][OUTIMROW][OUTIMROW],
   float Cin[NUM][INIMROW][INIMROW],
   float weight[NUM][NUM][KERNEL][KERNEL],
   float bias[NUM]
){
// please add your OpenCL setup code below
struct timeval t1, t2;
float elapsed_time;

static float C[NUM][IMROW][IMROW];

// Convolution is parallelized because it is the bottle neck of the computation speed
// use this to check the output of each API call
cl_int status;

/************************** PLATFORM **********************************/
// retrieve the number of platforms
cl_uint numPlatforms = 0;
status = clGetPlatformIDs(0, NULL, &numPlatforms);
checkErr(status, "Retrieve the number of platforms");
printf("Num of Platforms: %d\n", numPlatforms);

// Allocate enough space for each platform
cl_platform_id *platforms = NULL;
platforms = (cl_platform_id*)malloc(numPlatforms * sizeof(cl_platform_id));

// Fill in the platforms
status = clGetPlatformIDs(numPlatforms, platforms, NULL);
checkErr(status, "Fill in the platforms");

// Find GPU
int platform_index = -1;
int i;
for (i = 0; i < numPlatforms; i++)
{
    char vendor[128];
    clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, sizeof(vendor), vendor, NULL);
    char vendorF[7];
    memcpy((void*)vendorF, (void*)vendor, 6);
    vendorF[6] = '\0';
    fprintf(stderr, "%s\n", vendorF);
    if (strcmp(vendorF, "NVIDIA") == 0)
    {
        platform_index = i;
        break;
    }

}
if (platform_index == -1) {
    printf("GPU Platform not found!\n");
    exit(1);
}

/***************************** DEVICE *******************************/

// Retrieve the number of devices
cl_uint numDevices = 0;
status = clGetDeviceIDs(platforms[platform_index], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
checkErr(status, "Retrieve the number of devices");
printf("Number of devices: %d, status: %d\n", numDevices, status);

// Allocate enough space for each device
cl_device_id *devices;
devices = (cl_device_id*) malloc(numDevices * sizeof(cl_device_id));

// Fill in the devices
status = clGetDeviceIDs(platforms[platform_index], CL_DEVICE_TYPE_ALL, numDevices, devices, NULL);
checkErr(status, "Fill in the devices");

/*
char buffer[10240];
cl_uint buf_uint;
cl_ulong buf_ulong;
status = clGetDeviceInfo(devices[0], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(buf_uint), &buf_uint, NULL);
printf("DEVICE_MAX_WORK_GROUP_SIZE = %u\n", (unsigned int)buf_uint);
status = clGetDeviceInfo(devices[0], CL_DEVICE_ADDRESS_BITS, sizeof(buf_uint), &buf_uint, NULL);
printf("DEVICE ADDRESS BITS = %u\n", (unsigned int)buf_uint);
status = clGetDeviceInfo(devices[0], CL_DEVICE_NAME, sizeof(buffer), buffer, NULL);
printf("DEVICE_NAME = %s\n", buffer);
status = clGetDeviceInfo(devices[0], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(buf_ulong), &buf_ulong, NULL);
printf("DEVICE_LOCAL_MEM_SIZE = %llu\n", buf_ulong);
status = clGetDeviceInfo(devices[0], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(buf_uint), &buf_uint, NULL);
printf("DEVICE_MAX_WORK_ITEM_DIMENSION = %u\n", (unsigned int)buf_uint);
*/

/**************************** CONTEXT *******************************/

// create a context and associate it with the devices
cl_context context;
context = clCreateContext(NULL, numDevices, devices, NULL, NULL, &status);
checkErr(status, "Create context");

/**************************** COMMAND QUEUE *************************/

// create a command queue and associate it with the devices
cl_command_queue cmdQueue;
cmdQueue = clCreateCommandQueue(context, devices[0], 0, &status);
checkErr(status, "Create Command Queue");

/***************************** BUFFER ******************************/
// create a buffer object that will contain the data from the host
// Cin
cl_mem bufCin;
bufCin = clCreateBuffer(context, CL_MEM_READ_ONLY, NUM*INIMROW*INIMROW*sizeof(float), NULL, &status);

// Weight
cl_mem bufW;
bufW = clCreateBuffer(context, CL_MEM_READ_ONLY, NUM*NUM*KERNEL*KERNEL*sizeof(float), NULL, &status);

// bias
cl_mem bufBias;
bufBias = clCreateBuffer(context, CL_MEM_READ_ONLY, NUM*sizeof(float), NULL, &status);

// Cconv
cl_mem bufCconv;
bufCconv = clCreateBuffer(context, CL_MEM_READ_WRITE, NUM*IMROW*IMROW*sizeof(float), NULL, &status);

/****************************** ENQUEUE ****************************/

// Write input Cin to the device buffer bufCin
status = clEnqueueWriteBuffer(cmdQueue, bufCin, CL_FALSE, 0, NUM*INIMROW*INIMROW*sizeof(float),
    Cin, 0, NULL, NULL);
checkErr(status, "Write buffer Cin");

// Write input Weight to the device buffer bufW
status = clEnqueueWriteBuffer(cmdQueue, bufW, CL_FALSE, 0, NUM*NUM*KERNEL*KERNEL*sizeof(float),
    weight, 0, NULL, NULL);
checkErr(status, "Write buffer weight");

status = clEnqueueWriteBuffer(cmdQueue, bufBias, CL_FALSE, 0, NUM*sizeof(float), bias, 0, NULL, NULL);
checkErr(status, "Write buffer bias");

// Write conv buffer to the device buffer bufCconv
status = clEnqueueWriteBuffer(cmdQueue, bufCconv, CL_FALSE, 0, NUM*IMROW*IMROW*sizeof(float),
    C, 0, NULL, NULL);
checkErr(status, "Write buffer Cconv");

/****************************** PROGRAM ****************************/
// Create a program with source code
gettimeofday(&t1, NULL);

cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernel_cl, NULL, &status);

// Build the program for the device
status = clBuildProgram(program, numDevices, devices, NULL, NULL, NULL);
if (status == CL_BUILD_PROGRAM_FAILURE) {
    size_t log_size;
    clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    char *log = (char*)malloc(log_size);
    clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
    fprintf(stderr, "%s\n", log);
    exit(1);
}

gettimeofday(&t2, NULL);
elapsed_time = (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1e6;
fprintf(stderr, "Compilation time(s): %f\n", elapsed_time);

/***************************** KERNEL ************************************/
// create the conv kernel
cl_kernel kernel;
kernel = clCreateKernel(program, "conv", &status);

// Associate the input and output buffers with the kernel
status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufCin);
checkErr(status, "Set Arg 0");
status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufW);
checkErr(status, "Set Arg 1");
status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufBias);
checkErr(status, "Set Arg 2");
status = clSetKernelArg(kernel, 3, sizeof(cl_mem), &bufCconv);

/******************************** WORK SIZE *******************************/
// Define an index space of work items for execution. A workgroup size (local work size)
// is not required, but can be used.
size_t localWorkSize[3] = {1, BLOCK_SIZE, BLOCK_SIZE};
size_t globalWorkSize[3] = {NUM, IMROW, IMROW};


/******************************** EXECUTION *********************************/
gettimeofday(&t1, NULL);

// Execute the kernel
status = clEnqueueNDRangeKernel(cmdQueue, kernel, 3, NULL,
    globalWorkSize, localWorkSize, 0, NULL, NULL);
checkErr(status, "Execute Kernel");

// read the device output buffer
clEnqueueReadBuffer(cmdQueue, bufCconv, CL_TRUE, 0, NUM*IMROW*IMROW*sizeof(float),
    C, 0, NULL, NULL);

gettimeofday(&t2, NULL);
elapsed_time = (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1e6;
fprintf(stderr, "Convolution time(s): %f\n", elapsed_time);
fprintf(stderr, "Convolution GOPs: %f\n", (float)NUM * NUM * IMROW * IMROW * KERNEL * KERNEL * 2 / elapsed_time / 1e9);

clReleaseKernel(kernel);
clReleaseProgram(program);
clReleaseCommandQueue(cmdQueue);
clReleaseMemObject(bufCin);
clReleaseMemObject(bufW);
clReleaseMemObject(bufCconv);
clReleaseContext(context);

// end of convolution

// Max pooling
for (int i = 0; i < NUM; i++) {
    for (int h = 0; h < OUTIMROW; h++) {
        for (int w = 0; w < OUTIMROW; w++) {
            float local_max = C[i][2 * h][2 * w];
            local_max = fmax(local_max, C[i][2 * h + 1][2 * w]);
            local_max = fmax(local_max, C[i][2 * h + 1][2 * w + 1]);
            local_max = fmax(local_max, C[i][2 * h][2 * w + 1]);
            Cout[i][h][w] = local_max;
        }
    }
}
}

int main(){
	static float Cout[NUM][OUTIMROW][OUTIMROW];
	static float Cin[NUM][INIMROW][INIMROW];
	static float weight[NUM][NUM][KERNEL][KERNEL];
	static float bias[NUM];

	LoadData(Cin, weight, bias);

	fprintf(stderr, "Start cnn computation\n");
	struct timeval t1, t2;
	gettimeofday(&t1, NULL);

	// --- Please add OpenCL setup code inside the function below ---
   opencl_gpu(
      Cout, Cin, weight, bias
   );

   // --- Timing stuff
	gettimeofday(&t2, NULL);
	float elapsed_time = (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1e6;
	fprintf(stderr, "time(s): %f\n", elapsed_time);
	fprintf(stderr, "GOPs: %f\n", (float)NUM * NUM * IMROW * IMROW * KERNEL * KERNEL * 2 / elapsed_time / 1e9);

   // Please disable the error check before handing in your submission
   // Reminder: We will be measuring your performance externally! (using a unix time invocation)
	int error = Verify(Cout);
	if(error != 0)
		fprintf(stderr, "error ocurrs %d\n", error);
	else
		fprintf(stderr, "all right!\n");

	return 0;
}
