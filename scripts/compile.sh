set -euo pipefail

if [ "${1:-}" = "" ]; then
  echo "usage: source scripts/compile.sh <repo-root>" >&2
  return 2 2>/dev/null || exit 2
fi

ROOT="$1"

# Compiler settings (choose one):
# A) generic GCC
CC=$(which gcc)
CXX=$(which g++)
# B) possible alternative on AMD systems:
# CC=$(which hipcc)
# CXX=$(which hipcc)
# C) possible alternative on Cray systems:
# CC=$(which cc)
# CXX=$(which CC)
# D) possible alternative on Intel systems:
# CC=$(which icx)
# CXX=$(which icpx)
#
#
ulimit -s 131072
######################
#### compile demo ####
######################
export LD_LIBRARY_PATH="$ROOT/build/:${LD_LIBRARY_PATH:-}"
cd build
cmake -DCMAKE_INSTALL_PREFIX="$ROOT" -DCMAKE_CXX_STANDARD=17 -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" -DCMAKE_CXX_FLAGS="-g" ..
# make VERBOSE=1
make -j32
cd ..