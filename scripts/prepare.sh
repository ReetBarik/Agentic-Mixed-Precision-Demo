#!/bin/sh
# Set up the build environment.
# On JLSE (Argonne) this loads environment modules.
# On systems without the module system (e.g. local WSL2) it skips module loads
# gracefully — the compilers and cmake are expected to already be on PATH.

if command -v module >/dev/null 2>&1; then
    module use /soft/modulefiles
    module load gcc/13.3.0
    module load cmake/3.28.3

    # For NVIDIA
    module load cuda/12.9.1

    # For AMD
    # module load rocm/7.0.2

    # For Intel

    module list
fi
