#!/bin/sh
# Runs INSIDE a node. Resolves an example name to a CMake target, compiles it
# into the shared build volume, and reports the binary name.
#
#   sh /workspace/scripts/node/build-example.sh <name>
#   MPILAB_NO_COMPILE=1 sh /workspace/scripts/node/build-example.sh <name>
#
# Only one node ever runs this. The output lands in /build, which every node
# mounts, so the binary mpirun launches is literally the same file on every
# rank -- no copying, and no way for the nodes to drift out of sync.
#
# Configure happens once; after that this is an incremental build, which is
# what keeps the edit-run cycle in the seconds range.
#
# The last line of stdout is `mpilab-target=<name>`, which is how the caller
# learns what to launch when the directory and the binary are spelled
# differently. Everything else this prints is compiler output.

set -eu

NAME=${1:-}
SRC=${MPILAB_SRC_DIR:-/workspace}
BUILD=${MPILAB_BUILD_DIR:-/build}

[ -n "$NAME" ] || { echo "build-example: no example name given" >&2; exit 2; }

if [ ! -f "$BUILD/CMakeCache.txt" ]; then
    echo "[build] configuring $SRC -> $BUILD"
    cmake -S "$SRC" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release
fi

# `all` compiles every example. Used by tests/cluster/run_all.sh so the checks
# that follow can launch with --no-build without each one paying for a build.
if [ "$NAME" = "all" ]; then
    cmake --build "$BUILD" --parallel "$(nproc)"
    echo "mpilab-target=all"
    exit 0
fi

# examples.map is written by the root CMakeLists at configure time.
TARGET=$NAME
if [ -f "$BUILD/examples.map" ]; then
    resolved=$(sed -n "s/^${NAME}=//p" "$BUILD/examples.map" | head -1)
    if [ -n "$resolved" ]; then
        TARGET=$resolved
    else
        echo "build-example: '$NAME' is not an example. Known names:" >&2
        sed -n 's/^\([^#][^=]*\)=.*/  \1/p' "$BUILD/examples.map" | sort -u >&2
        exit 2
    fi
fi

if [ "${MPILAB_NO_COMPILE:-0}" != "1" ]; then
    cmake --build "$BUILD" --target "$TARGET" --parallel "$(nproc)"
fi

echo "mpilab-target=$TARGET"
