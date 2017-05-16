#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>
#include <string.h>
#include "cnn.h"

// OpenCL includes
#include <CL/cl.h>


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

inline void checkErr(cl_int err, const char * name) {
	if (err != CL_SUCCESS) {
		fprintf(stderr, "ERROR: %s (%d)\n", name, err);
		exit(EXIT_FAILURE);
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
	// conv(Cout, Cin, weight, bias);	
	
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
	//
	

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
