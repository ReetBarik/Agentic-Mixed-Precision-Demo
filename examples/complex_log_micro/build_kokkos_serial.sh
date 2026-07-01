#!/usr/bin/env bash
# Builds Kokkos with Serial backend only and installs to $HOME/kokkos-install.
# No GPU backend, no arch flags needed — runs on any x86 CPU.
#
# Usage:
#   bash examples/cln_micro/build_kokkos_serial.sh
#
# After this completes, build complex_log_micro with:
#   cd examples/cln_micro
#   cmake -B build -DKokkos_DIR=$HOME/kokkos-install/lib64/cmake/Kokkos
#   cmake --build build -j

set -euo pipefail

INSTALL_PREFIX="$HOME/kokkos-install"
KOKKOS_TAG=5.1.0
KOKKOS_URL=https://github.com/kokkos/kokkos.git
BUILD_DIR=$(mktemp -d)

echo "Cloning Kokkos $KOKKOS_TAG into $BUILD_DIR ..."
git clone --depth 1 "$KOKKOS_URL" -b "$KOKKOS_TAG" "$BUILD_DIR/kokkos"

echo "Configuring (Serial only, install → $INSTALL_PREFIX) ..."
cmake -S "$BUILD_DIR/kokkos" -B "$BUILD_DIR/build" \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_STANDARD=20 \
  -DCMAKE_CXX_EXTENSIONS=OFF \
  -DKokkos_ENABLE_SERIAL=ON

echo "Building and installing ..."
cmake --build "$BUILD_DIR/build" -j"$(nproc)" --target install

echo ""
echo "Done. Kokkos installed to $INSTALL_PREFIX"
echo ""
echo "To build complex_log_micro:"
echo "  cd examples/cln_micro"
echo "  cmake -B build -DKokkos_DIR=$INSTALL_PREFIX/lib64/cmake/Kokkos"
echo "  cmake --build build -j"
echo "  ./build/complex_log_micro"
