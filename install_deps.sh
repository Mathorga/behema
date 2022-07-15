#!/bin/bash

# Install Make.
sudo apt install make -y

# Install C compiler.
sudo apt install gcc -y

# Install simple image manipulation library. Handles pbm, pgm and ppm image file formats.
#sudo apt install netpbm

# Install OpenCL.
sudo apt install opencl-headers ocl-icd-opencl-dev -y

printf "\nDependencies installed!\n"
