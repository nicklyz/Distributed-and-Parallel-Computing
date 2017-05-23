#!/usr/bin/env bash

# Fetch CUDA
wget -O cuda.deb https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb

# Install CUDA
sudo dpkg -i cuda.deb
sudo apt-get update
sudo apt-get -y install cuda

# Install OpenCL
sudo apt-get install -y ocl-icd-libopencl1 opencl-headers clinfo
