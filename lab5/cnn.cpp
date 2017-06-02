#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <assert.h>
#include <stdbool.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <CL/opencl.h>
#include <time.h>
#include <sys/time.h>
#include "cnn.h"

int launch_kernel( 
    cl_command_queue &commands,          // compute command queue
    cl_kernel &kernel                    // compute kernel
    )
{
/**
 * Below is the template code for this lab.
 * You should modify the global/local work size and the kernel.cl to optimize your design.
 */

// Step 1 Define the global/local work size
    size_t local_size[2] = {1,1};
    size_t global_size[2] = {1,1};    

// Step 2 Enqueue the kernel
    int err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL,
                (size_t*)&global_size, (size_t*)&local_size, 0, NULL, NULL);
    if (err)
    {
      printf("Error: Failed to execute kernel! %d\n", err);
      printf("Test failed\n");
      return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

int ocl_fpga(
    float Cout[NUM][OUTIMROW][OUTIMROW],
    float Cin[NUM][INIMROW][INIMROW],
    float weight[NUM][NUM][KERNEL][KERNEL],
    float bias[NUM],
    char** argv
    )
{
#if defined(SDX_PLATFORM) && !defined(TARGET_DEVICE)
  #define STR_VALUE(arg)      #arg
  #define GET_STRING(name) STR_VALUE(name)
  #define TARGET_DEVICE GET_STRING(SDX_PLATFORM)
#endif
    //TARGET_DEVICE macro needs to be passed from gcc command line
    const char *target_device_name = TARGET_DEVICE;
    int err;                            // error code returned from api calls
    
    cl_platform_id platforms[16];       // platform id
    cl_platform_id platform_id;         // platform id
    cl_uint platform_count;
    cl_device_id device_id;             // compute device id 
    cl_context context;                 // compute context
    cl_command_queue commands;          // compute command queue
    cl_program program;                 // compute program
    cl_kernel kernel;                   // compute kernel
   
    char cl_platform_vendor[1001];
   
    cl_mem input_cin;                     // device memory used for the input array
    cl_mem input_bias;                    // device memory used for the input array
    cl_mem input_weight;                  // device memory used for the input array
    cl_mem output_cout;                        // device memory used for the output array
   

    // 
    // Get all platforms and then select Xilinx platform
    err = clGetPlatformIDs(16, platforms, &platform_count);
    if (err != CL_SUCCESS)
        {
            printf("Error: Failed to find an OpenCL platform!\n");
            printf("Test failed\n");
            return EXIT_FAILURE;
        }
    printf("INFO: Found %d platforms\n", platform_count);

    // Find Xilinx Plaftorm
    int platform_found = 0;
    for (unsigned int iplat=0; iplat<platform_count; iplat++) {
        err = clGetPlatformInfo(platforms[iplat], CL_PLATFORM_VENDOR, 1000, (void *)cl_platform_vendor,NULL);
        if (err != CL_SUCCESS) {
            printf("Error: clGetPlatformInfo(CL_PLATFORM_VENDOR) failed!\n");
            printf("Test failed\n");
            return EXIT_FAILURE;
        }
        if (strcmp(cl_platform_vendor, "Xilinx") == 0) {
            printf("INFO: Selected platform %d from %s\n", iplat, cl_platform_vendor);
            platform_id = platforms[iplat];
            platform_found = 1;
        }
    }
    if (!platform_found) {
        printf("ERROR: Platform Xilinx not found. Exit.\n");
        return EXIT_FAILURE;
    }
  
    // Connect to a compute device
    // find all devices and then select the target device
    cl_device_id devices[16];  // compute device id 
    cl_uint device_count;
    unsigned int device_found = 0;
    char cl_device_name[1001];
    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ACCELERATOR,
                         16, devices, &device_count);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to create a device group!\n");
        printf("Test failed\n");
        return EXIT_FAILURE;
    }

    //iterate all devices to select the target device. 
    for (int i=0; i<(int)device_count; i++) {
        err = clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 1024, cl_device_name, 0);
        if (err != CL_SUCCESS) {
            printf("Error: Failed to get device name for device %d!\n", i);
            printf("Test failed\n");
            return EXIT_FAILURE;
        }
        printf("INFO: Found device %s\n", cl_device_name);
        if(strcmp(cl_device_name, target_device_name) == 0) {
            device_id = devices[i];
            device_found = 1;
            printf("INFO: Selected %s as the target device\n", cl_device_name);
        }
    }
    
    if (!device_found) {
        printf("ERROR: Target device %s not found. Exit.\n", target_device_name);
        return EXIT_FAILURE;
    }

    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ACCELERATOR,
                         1, &device_id, NULL);
    if (err != CL_SUCCESS)
        {
            printf("Error: Failed to create a device group!\n");
            printf("Test failed\n");
            return EXIT_FAILURE;
        }
  
    // Create a compute context 
    //
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context)
        {
            printf("Error: Failed to create a compute context!\n");
            printf("Test failed\n");
            return EXIT_FAILURE;
        }

    // Create a command commands
    //
    commands = clCreateCommandQueue(context, device_id, 0, &err);
    if (!commands)
        {
            printf("Error: Failed to create a command commands!\n");
            printf("Error: code %i\n",err);
            printf("Test failed\n");
            return EXIT_FAILURE;
        }

    int status;

    // Load binary from disk
    unsigned char *kernelbinary;
    char *xclbin=argv[1];
    printf("INFO: Loading %s\n", xclbin);
    int n_i = load_file_to_memory(xclbin, (char **) &kernelbinary);
    if (n_i < 0) {
        printf("failed to load kernel from xclbin: %s\n", xclbin);
        printf("Test failed\n");
        return EXIT_FAILURE;
    }
    size_t n = n_i;
    // Create the compute program from offline
    program = clCreateProgramWithBinary(context, 1, &device_id, &n,
                                        (const unsigned char **) &kernelbinary, &status, &err);
    if ((!program) || (err!=CL_SUCCESS)) {
        printf("Error: Failed to create compute program from binary %d!\n", err);
        printf("Test failed\n");
        return EXIT_FAILURE;
    }

    // Build the program executable
    //
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
        {
            size_t len;
            char buffer[2048];

            printf("Error: Failed to build program executable!\n");
            clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
            printf("%s\n", buffer);
            printf("Test failed\n");
            return EXIT_FAILURE;
        }

    // Create the compute kernel in the program we wish to run
    //
    kernel = clCreateKernel(program, "cnn_kernel", &err);
    if (!kernel || err != CL_SUCCESS)
        {
            printf("Error: Failed to create compute kernel!\n");
            printf("Test failed\n");
            return EXIT_FAILURE;
        }

    // Create the input and output arrays in device memory for our calculation
    //
    input_cin = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * NUM * INIMROW * INIMROW, NULL, NULL);
	input_weight = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * NUM * NUM * KERNEL * KERNEL, NULL, NULL);
	input_bias = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * NUM, NULL, NULL);
	output_cout = clCreateBuffer(context,  CL_MEM_WRITE_ONLY,  sizeof(float) * NUM * OUTIMROW * OUTIMROW, NULL, NULL);
	
    if (!input_cin || !input_weight || !input_bias || !output_cout)
        {
            printf("Error: Failed to allocate device memory!\n");
            printf("Test failed\n");
            return EXIT_FAILURE;
        }    
    
    // Write our data set into the input array in device memory 
    //
    err = clEnqueueWriteBuffer(commands, input_cin, CL_TRUE, 0, sizeof(float) * NUM * INIMROW * INIMROW, Cin, 0, NULL, NULL);
    if (err != CL_SUCCESS)
        {
            printf("Error: Failed to write to source array Cin!\n");
            printf("Test failed\n");
            return EXIT_FAILURE;
        }

    // Write our data set into the input array in device memory 
    //
    err = clEnqueueWriteBuffer(commands, input_weight, CL_TRUE, 0, sizeof(float) * NUM * NUM * KERNEL * KERNEL, weight, 0, NULL, NULL);
    if (err != CL_SUCCESS)
        {
            printf("Error: Failed to write to source array weight!\n");
            printf("Test failed\n");
            return EXIT_FAILURE;
        }
		
    // Write our data set into the input array in device memory 
    //
    err = clEnqueueWriteBuffer(commands, input_bias, CL_TRUE, 0, sizeof(float) * NUM, bias, 0, NULL, NULL);
    if (err != CL_SUCCESS)
        {
            printf("Error: Failed to write to source array bias!\n");
            printf("Test failed\n");
            return EXIT_FAILURE;
        }		
    
    // Set the arguments to our compute kernel
    //
    err = 0;
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_cin);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &input_weight);
	  err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &input_bias);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &output_cout);
    if (err != CL_SUCCESS)
        {
            printf("Error: Failed to set kernel arguments! %d\n", err);
            printf("Test failed\n");
            return EXIT_FAILURE;
        }

    err = launch_kernel(commands, kernel);
    // Read back the results from the device to verify the output
    if (err == EXIT_FAILURE)
    {
      printf("Error: Failed to launch kernel! %d\n", err);
      printf("Test failed\n");
      return EXIT_FAILURE;
    }

    cl_event readevent;
    err = clEnqueueReadBuffer( commands, output_cout, CL_TRUE, 0, sizeof(int) * NUM * OUTIMROW * OUTIMROW, Cout, 0, NULL, &readevent );  
    if (err != CL_SUCCESS)
        {
            printf("Error: Failed to read output array! %d\n", err);
            printf("Test failed\n");
            return EXIT_FAILURE;
        }

    clWaitForEvents(1, &readevent);
    
    // Shutdown and cleanup
    //
    clReleaseMemObject(input_cin);
    clReleaseMemObject(input_weight);
  	clReleaseMemObject(input_bias);
    clReleaseMemObject(output_cout);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    return EXIT_SUCCESS;
}

int main(int argc, char** argv)
{
	static float Cout[NUM][OUTIMROW][OUTIMROW];
	static float Cin[NUM][INIMROW][INIMROW];
	static float weight[NUM][NUM][KERNEL][KERNEL];
	static float bias[NUM];
  if (argc != 2){
      printf("%s <inputfile>\n", argv[0]);
      return EXIT_FAILURE;
  }

	LoadData(Cin, weight, bias);

	// OpenCL host program
	fprintf(stderr, "Start cnn computation\n");
	struct timeval t1, t2;
	gettimeofday(&t1, NULL);
	
	int err = ocl_fpga(Cout, Cin, weight, bias, argv);	
  if (err == EXIT_FAILURE){
    return err;
  }

	gettimeofday(&t2, NULL);
	float elapsed_time = (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1e6;
	fprintf(stderr, "time(s): %f\n", elapsed_time);
	fprintf(stderr, "GOPs: %f\n", (float)NUM * NUM * IMROW * IMROW * KERNEL * KERNEL * 2 / elapsed_time / 1e9);

	int error = Verify(Cout);
	if(error != 0) {
		fprintf(stderr, "error ocurrs %d\n", error);
    return EXIT_FAILURE;
  }
	else {
		fprintf(stderr, "all right!\n");
    return EXIT_SUCCESS;
  }
}
