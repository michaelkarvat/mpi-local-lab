# syntax=docker/dockerfile:1
#
# The CPU image of the lab, in three stages that serve three different jobs.
#
#   builder   compiles the whole tree and runs the test suite. Not shipped.
#   dev       a simulated MPI node: toolchain, MPI, sshd. This is what
#             compose.yaml runs, four at a time, and what you develop in.
#   runtime   the object-matching example as a shipped artifact -- binary and
#             runtime libraries only, no compiler. Default target.
#
#   docker compose up -d                     # four dev nodes
#   docker build -t mpi-lab:cpu --target dev .
#   docker build -t msearch .                # runtime, the default
#
# CUDA is deliberately NOT here. A GPU image needs the NVIDIA driver on the
# host plus the NVIDIA Container Toolkit, which turns a one-command build into
# a host-configuration exercise. See Dockerfile.cuda and docs/CUDA.md.
#
# The dev stage is not a stripped-down runtime with a compiler added back. It
# carries no source at all: the repository arrives as a bind mount at
# /workspace and build output goes to a shared volume at /build, which is what
# makes an edit-compile-run cycle take seconds instead of an image rebuild.

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
COPY cmake/ cmake/
COPY examples/ examples/

RUN cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DMPILAB_ENABLE_CUDA=OFF \
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
# A simulated MPI node. Several of these on a private Docker network are the
# product; see compose.yaml.
FROM ubuntu:${UBUNTU_VERSION} AS dev

LABEL org.opencontainers.image.title="mpi-local-lab (dev node)" \
      org.opencontainers.image.description="Simulated MPI node: GCC, CMake, OpenMPI, OpenMP, sshd" \
      org.opencontainers.image.licenses="MIT"

ENV DEBIAN_FRONTEND=noninteractive
# openssh-server is what makes this a *node* rather than a container that
# happens to have MPI: mpirun starts remote processes over ssh, so without a
# listening sshd the other nodes are unreachable no matter how good the
# networking is. gdb, ping and ip are here because "reproduce an MPI bug
# locally" is a stated goal and all three are the first tools you reach for.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        libopenmpi-dev \
        openmpi-bin \
        openssh-server \
        python3 \
        gdb \
        iproute2 \
        iputils-ping \
        netcat-openbsd \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# The node setup -- user, ssh keys, sshd policy -- is shared with
# Dockerfile.cuda so the CPU and GPU nodes cannot drift apart. Read
# docker/setup-node.sh for why a private key is baked in and why UsePAM
# matters.
COPY docker/openmpi-mca-params.conf /etc/openmpi/openmpi-mca-params.conf
COPY docker/ssh_config /etc/mpi-lab-ssh_config
COPY docker/setup-node.sh /tmp/setup-node.sh
RUN sh /tmp/setup-node.sh && rm /tmp/setup-node.sh

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /workspace

# Documentation only -- compose.yaml deliberately publishes nothing, so port 22
# is reachable from the other nodes and from nowhere on the host.
EXPOSE 22

HEALTHCHECK --interval=5s --timeout=3s --start-period=2s --retries=10 \
    CMD nc -z 127.0.0.1 22 || exit 1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

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

COPY --from=builder /src/build/bin/msearch /usr/local/bin/msearch

# Flattened: the example's tests/data, tools and bench land at the working
# directory root so the commands in its README work verbatim inside the image.
WORKDIR /work
COPY examples/hybrid-object-matching/tests/data/ tests/data/
COPY examples/hybrid-object-matching/tools/ tools/
COPY examples/hybrid-object-matching/bench/ bench/
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
