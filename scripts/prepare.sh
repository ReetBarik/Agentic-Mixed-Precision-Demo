#!/bin/sh

module use /soft/modulefiles
module load gcc/13.3.0
module load cmake/3.28.3

# For NVIDIA
module load cuda/12.9.1

# For AMD
# module load rocm/7.0.2

# For Intel

module list