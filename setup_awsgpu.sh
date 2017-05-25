#!/bin/sh

export OPENCL_ROOT=/usr/local/cuda-8.0
export LD_LIBRARY_PATH=$OPENCL_ROOT/lib64:$LD_LIBRARY_PATH
export PATH=$OPENCL_ROOT/bin:$PATH
export PATH=/usr/local/cs/bin:$PATH
