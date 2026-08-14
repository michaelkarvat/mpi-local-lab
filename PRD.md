# PRD — Local MPI Development & Simulation Environment

## 1. Overview

Build a local development environment that allows developers and students to develop, run, debug, and experiment with MPI applications without requiring access to a physical HPC cluster or external servers.

The system will use Docker containers to simulate multiple MPI nodes on a single physical machine.

The project should support:

* MPI across multiple containers
* OpenMP inside each node
* Optional CUDA/GPU execution
* Repeatable local environments
* Example MPI applications
* Verification and benchmarking tools
* A simple developer workflow

The existing hybrid MPI/OpenMP/CUDA object-matching application will become an advanced example workload demonstrating the environment.

---

# 2. Problem

MPI applications are commonly developed and executed on Linux-based HPC clusters.

For developers who do not have access to such infrastructure, testing realistic MPI configurations can be inconvenient because they may need:

* multiple machines
* Linux environments
* MPI installations
* networking configuration
* SSH configuration
* SLURM or another scheduler
* GPU/CUDA environments

Running multiple MPI ranks on a single process host is useful, but it does not reproduce the structure of a multi-node environment.

The project should provide an easy way to simulate multiple MPI nodes locally using containers.

---

# 3. Product Goal

Allow a user to clone the repository and create a local MPI development cluster with minimal setup.

The desired experience should be approximately:

```bash
docker compose up -d
```

followed by something similar to:

```bash
./scripts/run-mpi.sh -n 4 examples/hello-mpi
```

The user should be able to run MPI processes across multiple Docker containers that communicate through a private Docker network.

---

# 4. Non-Goals

The project is not intended to:

* replace a real HPC cluster
* reproduce real network latency or bandwidth between physical nodes
* provide accurate multi-node scaling benchmarks
* replace SLURM
* emulate multiple physical GPUs
* provide cloud infrastructure management
* become a Kubernetes-based orchestration platform

The system is primarily intended for:

* development
* experimentation
* education
* correctness testing
* architecture demonstrations

---

# 5. Target Users

## Primary users

### Students

Students learning:

* MPI
* OpenMP
* CUDA
* parallel programming
* distributed-memory systems

who may not have permanent access to university HPC infrastructure.

### Developers

Developers who want to:

* prototype MPI programs locally
* reproduce MPI bugs
* test different rank configurations
* develop before deploying to a real cluster

### Recruiters / Engineers reviewing the project

The repository should demonstrate knowledge of:

* distributed systems
* MPI
* Docker
* Linux
* networking
* OpenMP
* CUDA
* build systems
* testing
* performance engineering

---

# 6. Core Architecture

The environment will use one reusable Docker image.

Multiple containers will be instantiated from that image.

```text
                    Host Machine
                         |
                    Docker Engine
                         |
                 Private MPI Network
                         |
        +----------------+----------------+
        |                |                |
     mpi-node-1       mpi-node-2       mpi-node-3       mpi-node-4
      Container        Container        Container        Container
         |                |                |                |
       Rank 0           Rank 1           Rank 2           Rank 3
```

Each container represents a simulated MPI node.

Containers should have:

* the same application image
* separate hostnames
* MPI installed
* access to shared example source code or binaries
* network connectivity to the other nodes

---

# 7. Execution Model

The architecture should support two independent dimensions.

## Runtime

Determines where processes execute:

* Local single process
* Local MPI ranks
* Docker multi-container MPI simulation
* Real HPC cluster

## Compute Backend

Determines how computation occurs inside each rank:

* Serial
* OpenMP
* CUDA

This separation should allow configurations such as:

```text
MPI + Serial
MPI + OpenMP
MPI + CUDA
MPI + OpenMP + CUDA
```

without coupling the environment to a specific application.

---

# 8. Docker Images

## CPU Image

`Dockerfile`

Contains:

* Linux
* GCC
* CMake
* OpenMPI
* OpenMP runtime
* project tools

Supports:

* Serial
* OpenMP
* MPI

---

## CUDA Image

`Dockerfile.cuda`

Based on an official NVIDIA CUDA image.

Contains:

* CUDA Toolkit
* GCC
* CMake
* OpenMPI
* OpenMP
* project tools

Supports:

* Serial
* OpenMP
* MPI
* CUDA

On Windows, GPU execution should work through:

```text
Windows
   ↓
WSL2
   ↓
Docker Desktop
   ↓
Linux CUDA Container
   ↓
NVIDIA GPU
```

The host should only require:

* supported NVIDIA GPU
* NVIDIA driver
* WSL2
* Docker Desktop

The CUDA Toolkit should live inside the container.

---

# 9. Local MPI Cluster Simulation

Use Docker Compose to create multiple MPI nodes.

Example:

```yaml
services:
  node1:
  node2:
  node3:
  node4:
```

All nodes should use the same image.

They should communicate over a private Docker network.

The configuration should handle any MPI requirements such as:

* host discovery
* SSH
* user permissions
* MPI runtime configuration

without requiring manual key exchange from the user.

---

# 10. Developer CLI / Scripts

The repository should provide simple helper commands.

Examples:

```bash
./scripts/start-cluster.sh
```

```bash
./scripts/stop-cluster.sh
```

```bash
./scripts/status.sh
```

```bash
./scripts/run-mpi.sh -n 4 examples/hello-mpi
```

Potential future command:

```bash
./mpi-lab run --nodes 4 --ranks 8 examples/distributed-sum
```

The first version does not require a custom CLI if shell scripts are sufficient.

Avoid unnecessary abstraction.

---

# 11. Example Applications

Examples should progress from simple to advanced.

## Example 1 — Hello MPI

Each rank prints:

```text
Hello from rank 0 of 4
Hello from rank 1 of 4
...
```

Purpose:

* verify MPI communication
* demonstrate rank assignment

---

## Example 2 — Ping Pong

Two ranks exchange messages.

Purpose:

* demonstrate point-to-point MPI communication
* show `MPI_Send` / `MPI_Recv`

---

## Example 3 — Distributed Sum

Split an array between ranks.

Each rank calculates a partial result.

Results are combined using:

```text
MPI_Reduce
```

Purpose:

* demonstrate distributed computation

---

## Example 4 — Matrix / Parallel Workload

A more computationally intensive MPI example.

Purpose:

* demonstrate work distribution
* enable basic timing comparisons

---

## Example 5 — Hybrid Object Matching

Move the existing project here as the advanced demonstration.

Technologies:

* MPI
* OpenMP
* CUDA
* deterministic verification
* CPU/GPU backend comparison
* performance benchmarking

This example demonstrates a realistic hybrid HPC workload.

---

# 12. Verification

The system should include automated correctness checks.

Examples should be tested using:

* single-process execution
* multiple MPI ranks
* multiple Docker containers

Where possible, results should be deterministic.

For the object-matching workload:

```text
Serial result
      =
OpenMP result
      =
CUDA result
      =
MPI result
```

---

# 13. Testing

Tests should cover:

## Environment tests

* Docker image builds
* containers start successfully
* MPI nodes can resolve each other
* MPI processes can communicate across containers

## MPI tests

Run configurations such as:

```text
1 rank
2 ranks
4 ranks
```

and ensure expected output.

## GPU tests

When an NVIDIA GPU is available:

* CUDA backend detected
* GPU execution succeeds
* CUDA output matches reference output

GPU tests should gracefully skip when no GPU exists.

---

# 14. CI

GitHub Actions should verify what can reasonably run on standard CI infrastructure.

CPU CI:

* build CPU image
* compile examples
* run unit tests
* run OpenMP tests
* run MPI tests
* start multi-container environment if practical

CUDA CI:

* compile CUDA code without requiring a GPU

Actual CUDA runtime testing may remain a local/manual verification unless a GPU CI runner is available.

---

# 15. Repository Structure

Target structure:

```text
mpi-local-lab/
│
├── docker/
│   └── optional Docker support files
│
├── Dockerfile
├── Dockerfile.cuda
├── compose.yaml
│
├── scripts/
│   ├── start-cluster.sh
│   ├── stop-cluster.sh
│   ├── status.sh
│   └── run-mpi.sh
│
├── examples/
│   ├── hello-mpi/
│   ├── ping-pong/
│   ├── distributed-sum/
│   └── hybrid-object-matching/
│
├── tests/
│
├── docs/
│   ├── ARCHITECTURE.md
│   ├── MPI_GUIDE.md
│   ├── CUDA.md
│   └── REAL_CLUSTER.md
│
├── CMakeLists.txt
└── README.md
```

---

# 16. README Goals

A user visiting the repository should understand within a few seconds:

> Run and experiment with MPI locally using Docker containers — no HPC cluster required.

The README should include:

* what the project does
* architecture diagram
* quick start
* prerequisites
* first MPI example
* multi-container simulation
* CUDA support
* examples
* limitations
* difference between simulated and real clusters
* deployment to a real HPC environment

---

# 17. Quick Start Target

The project should aim for an experience similar to:

```bash
git clone ...
cd mpi-local-lab

docker compose up -d

./scripts/run-mpi.sh -n 4 examples/hello-mpi
```

Expected result:

```text
Hello from rank 0 of 4
Hello from rank 1 of 4
Hello from rank 2 of 4
Hello from rank 3 of 4
```

Stopping:

```bash
docker compose down
```

---

# 18. CUDA Experience

For users with an NVIDIA GPU:

```bash
docker build -f Dockerfile.cuda -t mpi-lab:cuda .
```

Validate GPU access:

```bash
docker run --rm --gpus all mpi-lab:cuda nvidia-smi
```

Run a CUDA-enabled example:

```bash
docker run --rm --gpus all mpi-lab:cuda ...
```

Eventually, GPU-enabled containers may also participate in the simulated MPI cluster.

---

# 19. Important Limitation

The Docker MPI environment simulates **logical nodes**, not physical machines.

For example:

```text
node1
node2
node3
node4
```

may be four containers, but they still share:

* the same CPU
* the same RAM
* the same physical network adapter
* the same storage
* the same GPU hardware

Therefore:

* correctness experiments are meaningful
* architecture experiments are meaningful
* MPI communication behavior can be studied
* true multi-node performance scaling cannot be measured accurately

Performance claims involving multiple physical nodes should only come from a real cluster.

---

# 20. Success Criteria

Version 1 is successful when a new user can:

1. Clone the repository.
2. Build or pull the Docker environment.
3. Start multiple MPI containers locally.
4. Execute an MPI application across them.
5. Change the number of ranks/nodes.
6. Run included examples.
7. Verify results.
8. Optionally execute CUDA workloads using a local NVIDIA GPU.
9. Understand how the same application could later run on a real HPC cluster.

---

# 21. Migration from the Existing Project

Do not rewrite the existing codebase from scratch.

Reuse proven components where appropriate:

* Docker CPU image
* Docker CUDA image
* CMake infrastructure
* MPI configuration
* OpenMP support
* CUDA support
* testing infrastructure
* CI
* benchmark utilities

Refactor the repository so the **MPI local environment becomes the product** and the current hybrid object-matching application becomes an advanced example.

The migration should be incremental and performed on a separate branch.

Avoid large rewrites where existing tested components can be moved or generalized.

---

# 22. Future Extensions

Possible future features:

* configurable node count
* heterogeneous CPU/GPU nodes
* artificial network latency
* bandwidth throttling
* fault injection / node failure simulation
* MPI debugging tools
* web-based cluster visualization
* SLURM-like local job launcher
* multi-GPU support
* performance dashboard
* custom user workloads mounted into the cluster

These are future possibilities and should not be part of the initial implementation unless they provide clear value without significant complexity.
