# mpi-local-lab

**Run and experiment with MPI locally using Docker containers — no HPC cluster
required.**

Four Linux containers on a private Docker network, each behaving like a
separate MPI node: its own hostname, its own sshd, reachable from the others.
Write an MPI program on your laptop, run it across all four, change the node
count, and see which node each rank landed on.

[![CI](https://github.com/michaelkarvat/Hybrid-MPI-OpenMP-CUDA/actions/workflows/ci.yml/badge.svg)](../../actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

```console
$ docker compose up -d
$ ./scripts/run-mpi.sh -n 4 examples/hello-mpi
Hello from rank 0 of 4 on 9554245ca3e0
Hello from rank 1 of 4 on a63bf6a3d2a7
Hello from rank 2 of 4 on 1cb7287251cb
Hello from rank 3 of 4 on f6813f33c8e8
```

Four different hostnames. That is the whole idea: the ranks are not four
processes on one machine, they are four processes on four machines that MPI
believes are separate.

---

## Why

MPI is a distributed-memory model, and the interesting parts of it — host
discovery, process launch over ssh, collectives that cross a network, code that
behaves differently at 2 ranks than at 4 — only appear when there is more than
one machine. Learning it usually means getting an account on a cluster first.

Running `mpirun -n 4` on one host is a useful approximation, but it never
exercises any of that. Every rank shares an address space's worth of luck: the
same filesystem, the same hostname, shared-memory transports instead of TCP.
Programs that would deadlock on a real cluster run fine.

This repository gives you the structure of a multi-node environment on one
machine, with two commands and no accounts:

- **MPI across multiple containers**, launched over ssh, exactly as a real
  cluster does it
- **OpenMP inside each node**, so hybrid MPI+OpenMP is a configuration and not
  a rewrite
- **Optional CUDA**, with the toolkit inside the container and nothing but a
  driver on the host
- **Examples from four lines to four thousand**, ending in a real hybrid
  workload
- **Tests that check the environment**, not just the programs

It is a development and teaching environment. It is [explicitly not a
performance lab](#what-this-cannot-tell-you) — see the limitations, which are
not a footnote.

## Architecture

```text
                        Host Machine
                             │
                       Docker Engine
                             │
                   mpi-lab-net (bridge)
                             │
     ┌───────────────┬───────┴───────┬───────────────┐
     │               │               │               │
  node-1          node-2          node-3          node-4
 container       container       container       container
     │               │               │               │
  Rank 0          Rank 1          Rank 2          Rank 3
     └───────────────┴───────┬───────┴───────────────┘
                             │
              /workspace  ← your source, bind-mounted
              /build      ← shared volume, one compile for all
```

Two axes, kept independent, because they answer different questions:

| | |
|---|---|
| **Where do processes run?** | a **runtime** — one process · local MPI ranks · this container cluster · a real cluster |
| **How does one rank compute?** | a **backend** — serial · OpenMP · CUDA |

Any combination works, and nothing in an example has to know which one it got.
That separation is what lets the same source run under `mpirun -n 2` on your
laptop, across four containers here, and under SLURM on a real machine.

Full reasoning — including why ssh, why one replicated service instead of four
named ones, and what the containers do *not* simulate — is in
**[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

## Prerequisites

| | |
|---|---|
| **Docker Desktop** (or Docker Engine + Compose v2) | the only hard requirement |
| **Windows: WSL2** | Docker Desktop needs it anyway; `wsl --install` |
| **A POSIX shell** for `./scripts/*.sh` | Git Bash ships with Git; WSL works too |
| **Optional: NVIDIA GPU + driver** | for the CUDA examples — [docs/CUDA.md](docs/CUDA.md) |

No compiler, no CMake, no MPI, and no CUDA Toolkit on the host. Everything that
builds anything lives in the containers.

## Quick start

```bash
git clone https://github.com/michaelkarvat/Hybrid-MPI-OpenMP-CUDA.git mpi-local-lab
cd mpi-local-lab

docker compose up -d                          # four nodes
./scripts/run-mpi.sh -n 4 examples/hello-mpi
```

Expected:

```text
Hello from rank 0 of 4 on 9554245ca3e0
Hello from rank 1 of 4 on a63bf6a3d2a7
Hello from rank 2 of 4 on 1cb7287251cb
Hello from rank 3 of 4 on f6813f33c8e8
```

Those hostnames are container ids. To see which node is which:

```console
$ ./scripts/status.sh
NODE                HOSTNAME        IP             STATE      HEALTH
mpi-lab-node-1      9554245ca3e0    172.18.0.4     running    healthy
mpi-lab-node-2      a63bf6a3d2a7    172.18.0.2     running    healthy
mpi-lab-node-3      f6813f33c8e8    172.18.0.5     running    healthy
mpi-lab-node-4      1cb7287251cb    172.18.0.3     running    healthy

nodes: 4   network: mpi-lab-net   build volume: mpi-lab-build
```

Stop when you are done:

```bash
docker compose down
```

`./scripts/start-cluster.sh` does the same as `docker compose up -d` but waits
until every node's sshd is actually listening, which is what you want before a
script launches into it.

## The commands

| | |
|---|---|
| `./scripts/start-cluster.sh [N]` | start N nodes (default 4) and wait until they are ready |
| `./scripts/stop-cluster.sh [--purge]` | stop; `--purge` also deletes the shared build volume |
| `./scripts/status.sh` | node ↔ hostname ↔ IP, and health |
| `./scripts/run-mpi.sh -n R <example> [args]` | compile the example and run it on R ranks |
| `./scripts/shell.sh [N]` | interactive shell on node N |

`-n` is the number of **ranks**. The number of **nodes** is however many
containers are running. Ranks are placed round-robin across nodes, so `-n 4` on
a four-node cluster is one rank per node and `-n 8` is two.

### Changing the cluster size

No file to edit:

```console
$ ./scripts/start-cluster.sh 6
$ ./scripts/run-mpi.sh -n 6 examples/hello-mpi
Hello from rank 0 of 6 on 9554245ca3e0
Hello from rank 1 of 6 on a63bf6a3d2a7
Hello from rank 2 of 6 on 1cb7287251cb
Hello from rank 3 of 6 on f6813f33c8e8
Hello from rank 4 of 6 on fdb53a44fa8f
Hello from rank 5 of 6 on 1f34f6946755

$ ./scripts/run-mpi.sh -n 12 distributed-sum     # 12 ranks over 6 nodes
```

## Examples

Progressively harder, and each one exists to show a specific thing.

| | Example | What it demonstrates |
|---|---|---|
| 1 | [`hello-mpi`](examples/hello-mpi/) | rank ids, world size, and *which node* a rank landed on |
| 2 | [`ping-pong`](examples/ping-pong/) | `MPI_Send`/`MPI_Recv`, matching tags, round-trip latency |
| 3 | [`distributed-sum`](examples/distributed-sum/) | `MPI_Scatterv` + `MPI_Reduce` over an uneven split |
| 4 | [`matrix-multiply`](examples/matrix-multiply/) | row-block distribution, MPI **and** OpenMP together |
| 5 | [`hybrid-object-matching`](examples/hybrid-object-matching/) | MPI + OpenMP + CUDA, three interchangeable backends, byte-identical results |

```bash
./scripts/run-mpi.sh -n 2 ping-pong 1000 65536
./scripts/run-mpi.sh -n 4 distributed-sum 5000000
./scripts/run-mpi.sh -n 4 matrix-multiply 512
./scripts/run-mpi.sh -n 4 msearch --backend openmp \
    -i examples/hybrid-object-matching/tests/data/reference.txt -o -
```

Example 5 is the previous life of this repository, and it is worth reading on
its own: a submatrix search whose serial, OpenMP and CUDA backends are asserted
**byte-identical**, with a determinism contract that makes that assertion
meaningful. See
[its README](examples/hybrid-object-matching/README.md).

### Writing your own

Add a directory under `examples/`, a `CMakeLists.txt` of about four lines, and
you are done:

```cmake
mpilab_add_example(my-example
  SOURCES main.c
  MPI
  DESCRIPTION "what it shows")

mpilab_add_rank_test(my-example RANKS 1 2 4 MODE invariant)
```

`mpilab_add_rank_test` registers a check that the program produces the *same
output at every rank count* — which is the property most MPI bugs break first.
More in [docs/MPI_GUIDE.md](docs/MPI_GUIDE.md).

## CUDA

The CUDA Toolkit lives in the container. On the host you need a recent NVIDIA
driver and nothing else — not on Windows, not in your WSL distro.

```bash
docker build -f Dockerfile.cuda --target dev -t mpi-lab:cuda .
docker run --rm --gpus all mpi-lab:cuda nvidia-smi

./scripts/start-cluster.sh --gpu 2       # GPU nodes in the simulated cluster
./scripts/run-mpi.sh -n 1 msearch --verify \
    -i examples/hybrid-object-matching/tests/data/reference.txt
```

```console
reference: serial (10 pictures)
[info ] cuda backend: device 0 (NVIDIA GeForce RTX 3050 4GB Laptop GPU, 20 SMs)
  cuda     OK (identical to serial)
  openmp   OK (identical to serial)
```

Every GPU node maps onto the **same physical card**, so this is for correctness
work, not scaling. Setup, requirements and the caveats are in
[docs/CUDA.md](docs/CUDA.md).

## Testing

Two suites, checking two different things.

```bash
# Programs: unit tests, golden output, and rank sweeps -- inside one container
./scripts/shell.sh
  cmake -S /workspace -B /build && ctest --test-dir /build --output-on-failure

# Environment: the cluster itself -- from the host, against live containers
bash tests/cluster/run_all.sh
```

```console
$ bash tests/cluster/run_all.sh
0. build -- compile every example once, on node 1
  ok   every example built into the shared volume
1. name resolution -- every node can find every other node
  ok   node-1 resolves 'node' to 4 address(es)
  ...
3. cross-container MPI -- ranks land on distinct nodes
  ok   4 rank(s) reported
  ok   4 distinct hostname(s) -- the ranks really are on separate nodes
...
passed 15, failed 0
```

The second suite is the one that matters here. Everything in the first would
pass on a machine with no Docker at all.

CI runs both, plus builds with OpenMP off, MPI off and everything off, because
"it builds on my machine" is not the same claim as "it builds".

## What this cannot tell you

The containers are **logical nodes, not physical machines**. `node-1` through
`node-4` share:

- the same CPU and the same cores
- the same RAM and the same memory bandwidth
- the same physical network adapter — traffic between "nodes" never leaves the host
- the same disk
- the same GPU, if you use the GPU overlay

So:

| | |
|---|---|
| Correctness across ranks | **meaningful** — a collective that deadlocks here deadlocks on a cluster |
| Architecture and communication patterns | **meaningful** — this is real MPI over real TCP |
| Latency and bandwidth figures | **not meaningful** — you are measuring a loopback bridge |
| Multi-node scaling | **not meaningful** — adding nodes adds no hardware |

`ping-pong` will happily print a round-trip time. It is a real measurement of
this machine's virtual bridge and tells you nothing about an interconnect. Any
scaling claim involving multiple physical nodes has to come from a real
cluster.

This is a hard boundary of the approach, not a bug to be fixed later; see
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#what-is-not-simulated).

## Going to a real cluster

Nothing here is container-specific. The same source, the same `CMakeLists.txt`
and the same binaries run under SLURM:

```bash
scripts/slurm/build.sh                  # build once, on the login node
sbatch scripts/slurm/submit.sbatch
```

What changes, what does not, and which of your local assumptions will break, is
in [docs/REAL_CLUSTER.md](docs/REAL_CLUSTER.md).

## Repository layout

```text
compose.yaml            the cluster: one node service, scaled
compose.gpu.yaml        overlay that swaps in the CUDA image
Dockerfile              CPU:  builder → dev (a node) → runtime (an artifact)
Dockerfile.cuda         GPU:  the same three, on an NVIDIA base
docker/                 what makes a container a node: sshd, keys, MCA params
scripts/                start · stop · status · run-mpi · shell
  node/                 helpers that run inside a node
cmake/                  mpilab_add_example(), the rank-sweep harness
examples/               1 hello-mpi · 2 ping-pong · 3 distributed-sum
                        4 matrix-multiply · 5 hybrid-object-matching
tests/cluster/          environment tests, run against a live cluster
docs/                   ARCHITECTURE · MPI_GUIDE · CUDA · REAL_CLUSTER
```

## Technologies

C11 · MPI-3 · OpenMP · CUDA · Docker · Docker Compose · CMake · CTest · Bash ·
OpenSSH · GitHub Actions · SLURM

## License

MIT — see [LICENSE](LICENSE).
