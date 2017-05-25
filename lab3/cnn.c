#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>
#include <string.h>
#include "cnn.h"

// OpenCL includes
#include <CL/cl.h>
#include "kernel_cl.h"


inline void checkErr(cl_int err, const char * name) {
	if (err != CL_SUCCESS) {
		fprintf(stderr, "ERROR: %s (%d)\n", name, err);
		exit(EXIT_FAILURE);
	}
}

// Parallel CNN implementation
void conv(float Cout[NUM][OUTIMROW][OUTIMROW], float Cin[NUM][INIMROW][INIMROW],
          float weight[NUM][NUM][KERNEL][KERNEL], float bias[NUM])
{
	struct timeval t1, t2;
	float elapsed_time;

	static float C[NUM][IMROW][IMROW];

	for(int i = 0; i < NUM; i++) {
		for(int h = 0; h < IMROW; h++) {
			for(int w = 0; w < IMROW; w++)
				C[i][h][w] = bias[i];
		}
	}

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

	// Find CPU
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
		if (strcmp(vendorF, "Intel(") == 0)
		{
			platform_index = i;
			break;
		}

	}
	if (platform_index == -1) {
		printf("CPU Platform not found!\n");
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

	/**************************** CONTEXT *******************************/

	// create a context and associate it with the devices
	cl_context context;
	context = clCreateContext(NULL, numDevices, devices, NULL, NULL, &status);

	/**************************** COMMAND QUEUE *************************/
	
	// create a command queue and associate it with the devices
	cl_command_queue cmdQueue;
	cmdQueue = clCreateCommandQueue(context, devices[0], 0, &status);

	/***************************** BUFFER ******************************/
	// create a buffer object that will contain the data from the host
	// Cin
	cl_mem bufCin;
	bufCin = clCreateBuffer(context, CL_MEM_READ_ONLY, NUM*INIMROW*INIMROW*sizeof(float), NULL, &status);
	
	// Weight
	cl_mem bufW;
	bufW = clCreateBuffer(context, CL_MEM_READ_ONLY, NUM*NUM*KERNEL*KERNEL*sizeof(float), NULL, &status);

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
	status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufCconv);
	checkErr(status, "Set Arg 2");

	/******************************** WORK SIZE *******************************/
	// Define an index space of work items for execution. A workgroup size (local work size)
	// is not required, but can be used.
	size_t globalWorkSize[1];

	// There are NUM work-items
	globalWorkSize[0] = NUM;

	/******************************** EXECUTION *********************************/
	gettimeofday(&t1, NULL);

	// Execute the kernel
	status = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, NULL, 
		globalWorkSize, NULL, 0, NULL, NULL);
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

int main()
{
	static float Cout[NUM][OUTIMROW][OUTIMROW];
	static float Cin[NUM][INIMROW][INIMROW];
	static float weight[NUM][NUM][KERNEL][KERNEL];
	static float bias[NUM];

	LoadData(Cin, weight, bias);

	// OpenCL host program
	fprintf(stderr, "Start cnn computation\n");
	struct timeval t1, t2;
	gettimeofday(&t1, NULL);
	// --- Please add your code below ---
	conv(Cout, Cin, weight, bias);
	
	gettimeofday(&t2, NULL);
	float elapsed_time = (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1e6;
	fprintf(stderr, "time(s): %f\n", elapsed_time);
	fprintf(stderr, "GOPs: %f\n", (float)NUM * NUM * IMROW * IMROW * KERNEL * KERNEL * 2 / elapsed_time / 1e9);

	int error = Verify(Cout);
	if(error != 0)
		fprintf(stderr, "error ocurrs %d\n", error);
	else
		fprintf(stderr, "all right!\n");

	return 0;
}
