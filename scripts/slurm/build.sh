#!/bin/bash
# Build on a login node, before submitting. Keeping the build out of the batch
# job means a compile error surfaces in seconds instead of consuming an
# allocation, and repeated submissions do not recompile identical sources.
#
#   scripts/slurm/build.sh            # Release build with everything detected
#   scripts/slurm/build.sh Debug      # Debug build

set -euo pipefail

BUILD_TYPE=${1:-Release}
BUILD_DIR=${BUILD_DIR:-build}

# Adjust to your site's module names; these are the usual suspects.
module load cuda 2>/dev/null || echo "note: no 'cuda' module, relying on PATH"
module load openmpi 2>/dev/null || module load mpi 2>/dev/null || \
    echo "note: no MPI module, relying on PATH"

cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
cmake --build "$BUILD_DIR" --parallel

echo
"$BUILD_DIR/bin/msearch" --list-backends
echo
echo "Built $BUILD_DIR/bin/msearch. Submit with: sbatch scripts/slurm/submit.sbatch"
