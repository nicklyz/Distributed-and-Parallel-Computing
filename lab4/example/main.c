// This program implements a vector addition using OpenCL

// System includes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// OpenCL includes
#include <CL/cl.h>
#include "vecadd_cl.h"

#define GLOBAL_SIZE 1024 * 64
#define LOCAL_SIZE 64

inline void checkErr(cl_int err, const char * name) {
   if (err != CL_SUCCESS) {
      fprintf(stderr, "ERROR: %s (%d)\n", name, err);
      exit(EXIT_FAILURE);
   }
}

int main() {
   // This code executes on the OpenCL host

   // Host data
   int *A = NULL;  // Input array
   int *B = NULL;  // Input array
   int *C = NULL;  // Output array

   // Elements in each array
   const int elements = GLOBAL_SIZE;

   // Compute the size of the data 
   size_t datasize = sizeof(int) * elements;

   // Allocate space for input/output data
   A = (int*)malloc(datasize);
   B = (int*)malloc(datasize);
   C = (int*)malloc(datasize);

   // Initialize the input data
   int i;
   for(i = 0; i < elements; i++) {
      A[i] = i;
      B[i] = i;
   }

   // Use this to check the output of each API call
   cl_int status;  

   // Retrieve the number of platforms
   cl_uint numPlatforms = 0;
   status = clGetPlatformIDs(0, NULL, &numPlatforms);
   checkErr(status, "Retrieve the number of platforms");

   // Allocate enough space for each platform
   cl_platform_id *platforms = NULL;
   platforms = (cl_platform_id*)malloc(
         numPlatforms * sizeof(cl_platform_id));

   // Fill in the platforms
   status = clGetPlatformIDs(numPlatforms, platforms, NULL);
   checkErr(status, "Fill in the platforms");

   // Find CPU
   int platform_index = -1;
   for (i = 0; i < numPlatforms; i++){
      char vendor[128];
      clGetPlatformInfo (platforms[i], CL_PLATFORM_VENDOR, sizeof(vendor), vendor, NULL);
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
   if (platform_index == -1){
      printf("GPU platform not found!\n");
      exit(1);
   }

   // Retrieve the number of devices
   cl_uint numDevices = 0;
   status = clGetDeviceIDs(platforms[platform_index], CL_DEVICE_TYPE_ALL, 0, 
         NULL, &numDevices);
   checkErr(status, "Retrieve the number of devices");
   printf("#devices: %d, status %d\n", numDevices, status);

   // Allocate enough space for each device
   cl_device_id *devices;
   devices = (cl_device_id*)malloc(
         numDevices * sizeof(cl_device_id));

   // Fill in the devices 
   status = clGetDeviceIDs(platforms[platform_index], CL_DEVICE_TYPE_ALL,        
         numDevices, devices, NULL);
   checkErr(status, "Fill in the devices");

   // Create a context and associate it with the devices
   cl_context context;
   context = clCreateContext(NULL, numDevices, devices, NULL, 
         NULL, &status);

   // Create a command queue and associate it with the device 
   cl_command_queue cmdQueue;
   cmdQueue = clCreateCommandQueue(context, devices[0], 0, 
         &status);

   // Create a buffer object that will contain the data 
   // from the host array A
   cl_mem bufA;
   bufA = clCreateBuffer(context, CL_MEM_READ_ONLY, datasize,                       
         NULL, &status);

   // Create a buffer object that will contain the data 
   // from the host array B
   cl_mem bufB;
   bufB = clCreateBuffer(context, CL_MEM_READ_ONLY, datasize,                        
         NULL, &status);

   // Create a buffer object that will hold the output data
   cl_mem bufC;
   bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, datasize,
         NULL, &status); 

   // Write input array A to the device buffer bufferA
   status = clEnqueueWriteBuffer(cmdQueue, bufA, CL_FALSE, 
         0, datasize, A, 0, NULL, NULL);
   checkErr(status, "Write buffer A");

   // Write input array B to the device buffer bufferB
   status = clEnqueueWriteBuffer(cmdQueue, bufB, CL_FALSE, 
         0, datasize, B, 0, NULL, NULL);
   checkErr(status, "Write buffer B");

   // Create a program with source code
   cl_program program = clCreateProgramWithSource(context, 1, 
         (const char**)&vecadd_cl, NULL, &status);

   // Build (compile) the program for the device
   status = clBuildProgram(program, numDevices, devices, 
         NULL, NULL, NULL);
   if(status == CL_BUILD_PROGRAM_FAILURE) {
      size_t log_size;
      clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0,
            NULL, &log_size);
      char *log = (char*)malloc(log_size);
      clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG,
            log_size, log, NULL);
      fprintf(stderr, "%s\n", log);
      exit(1);
   }
   //		checkErr(status, "Build program");

   // Create the vector addition kernel
   cl_kernel kernel;
   kernel = clCreateKernel(program, "vecadd", &status);

   // Associate the input and output buffers with the kernel 
   status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
   checkErr(status, "Set Arg 0");
   status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
   checkErr(status, "Set Arg 1");
   status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);
   checkErr(status, "Set Arg 2");

   // Define an index space (global work size) of work 
   // items for execution. A workgroup size (local work size) 
   // is not required, but can be used.
   size_t local[1]  = {LOCAL_SIZE};
   size_t global[1] = {GLOBAL_SIZE};

   // Execute the kernel for execution
   status = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, NULL, 
         global, local, 0, NULL, NULL);
   checkErr(status, "Execute kernel");

   // Read the device output buffer to the host output array
   clEnqueueReadBuffer(cmdQueue, bufC, CL_TRUE, 0, 
         datasize, C, 0, NULL, NULL);

   // Verify the output
   int result = 1;
   for(i = 0; i < elements; i++) {
      if(C[i] != i + i) {
         result = 0;
         break;
      }
   }
   if(result) {
      printf("Output is correct\n");
   } else {
      printf("Output is incorrect\n");
   }

   // Free OpenCL resources
   clReleaseKernel(kernel);
   clReleaseProgram(program);
   clReleaseCommandQueue(cmdQueue);
   clReleaseMemObject(bufA);
   clReleaseMemObject(bufB);
   clReleaseMemObject(bufC);
   clReleaseContext(context);

   // Free host resources
   free(A);
   free(B);
   free(C);
   free(platforms);
   free(devices);

   return 0;
}

