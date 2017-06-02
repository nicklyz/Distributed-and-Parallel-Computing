OpenCL Matrix Multiplication CL Example
--------------------------------------------------------------------------------
This is an implementation of performing matrix multiplication of two 16x16 
matrices and getting the result back in 16x16 matrix. 

The main algorithm characteristics of this application are:

1. The matrices are flattened out in a single dimension before sending 
   it to FPGA, so the kernel operates the matrix multiplication under that assumption.
2. The example is parameterized, so it can be increased to larger size. 
3. This is a starter example to illustrate the use flow of SDAccel. Users may need 
   to modify their source code when multiplying very large matrices. 

Files in the Example
-------------------------------------------------------------------------------
Application host code
* test-cl.c

Kernel code
* mmult1.cl

Compilation File
* sdaccel.mk : Makefile for compiling SDAccel application

Compilation and Emulation
---------------------------
* Set up environment for SDAccel
* Run "make -f sdaccel.mk help" for further instructions

Executing the Application on FPGA
---------------------------------
* Run the application
  ./mmult_ex bin_mmult_hw.xclbin
