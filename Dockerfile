# syntax=docker/dockerfile:1
#
# Portable CPU image: serial + OpenMP backends, and a working MPI runtime.
#
# CUDA is deliberately NOT here. A GPU image needs the NVIDIA driver on the
# host plus the NVIDIA Container Toolkit, which turns a one-command build into
# a host-configuration exercise. The CUDA backend is built by the native CMake
# build (see README) and covered separately in CI. This image is the "clone and
# run it" path, not a replacement for building natively.
#
#   docker build -t msearch .
#   docker run --rm msearch msearch --list-backends
#
# Multi-stage: the compiler, headers and CMake stay in the builder, so the
# shipped image carries only the binary and the runtime libraries it links.

ARG UBUNTU_VERSION=24.04

# --------------------------------------------------------------------------
FROM ubuntu:${UBUNTU_VERSION} AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        libopenmpi-dev \
        openmpi-bin \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY CMakeLists.txt ./
COPY include/ include/
COPY src/ src/
COPY tests/ tests/

RUN cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DMSEARCH_ENABLE_CUDA=OFF \
    && cmake --build build --parallel "$(nproc)"

# The test suite runs during the build, so a broken commit cannot produce a
# usable image. OpenMPI refuses to run as root without explicit consent; the
# override lives only in this throwaway stage, never in the shipped image,
# which runs as an ordinary user.
ENV OMPI_ALLOW_RUN_AS_ROOT=1 \
    OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    OMPI_MCA_btl_vader_single_copy_mechanism=none
RUN ctest --test-dir build --output-on-failure

# --------------------------------------------------------------------------
FROM ubuntu:${UBUNTU_VERSION} AS runtime

LABEL org.opencontainers.image.title="hybrid-parallel-object-matching" \
      org.opencontainers.image.description="Parallel submatrix search (serial/OpenMP/MPI)" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/michaelkarvat/Hybrid-MPI-OpenMP-CUDA"

ENV DEBIAN_FRONTEND=noninteractive
# openmpi-bin brings mpirun and libmpi; libgomp1 is the OpenMP runtime;
# python3 makes tools/gen_input.py and bench/run_bench.py usable in here.
RUN apt-get update && apt-get install -y --no-install-recommends \
        openmpi-bin \
        libgomp1 \
        python3 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash msearch

COPY --from=builder /src/build/msearch /usr/local/bin/msearch

WORKDIR /work
COPY tests/data/ tests/data/
COPY tools/ tools/
COPY bench/ bench/
RUN chown -R msearch:msearch /work

# Unprivileged: mpirun refuses to run as root anyway, and nothing here needs it.
USER msearch

# Shared-memory transport inside a container falls back to a copy mechanism
# that needs CAP_SYS_PTRACE. Disabling it keeps `docker run` free of
# --cap-add flags at a negligible cost for a job of this shape.
ENV OMPI_MCA_btl_vader_single_copy_mechanism=none

# No ENTRYPOINT: the command is written out in full at the call site, so the
# same invocation works for msearch, mpirun and the Python tooling.
CMD ["msearch", "--help"]
