# CUDA

GPU execution with **no CUDA Toolkit on the host** — not on Windows, not in
your WSL distro. A recent NVIDIA driver is the only host requirement; `nvcc`,
the CUDA runtime and every library come from the image.

```text
Windows
   ↓
WSL2
   ↓
Docker Desktop
   ↓
Linux CUDA Container   ← nvcc and the CUDA runtime live here
   ↓
NVIDIA GPU
```

---

## What you need

### Windows

1. **Windows 10 (21H2+) or Windows 11**
2. **WSL2** — `wsl --install`, then `wsl --set-default-version 2`
3. **Docker Desktop** on the **WSL2 backend**
   (Settings → General → *Use the WSL 2 based engine*)
4. **A supported NVIDIA GPU** (Maxwell or newer; developed against an RTX 3050)
5. **An NVIDIA Windows driver with WSL2/CUDA support** — any recent one
   (≥ 527.41). Install it **on Windows**, not inside WSL, and do **not**
   install a Linux driver in the distro.

Nothing else. Docker Desktop supplies the container runtime; the image supplies
the toolkit.

### Linux

The NVIDIA driver plus the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/).

## Check the plumbing first

Before building anything in this repository:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

A table listing your GPU means you are ready. `could not select device driver`
or `unknown flag: --gpus` means step 3 or 5 above is not in place — fix that
first, because every failure after this point will look like a problem with
this project instead.

## One GPU node

The straightforward configuration, and the one to use for real GPU work.

```bash
docker build -f Dockerfile.cuda --target dev -t mpi-lab:cuda .

docker run --rm --gpus all -v "$PWD":/workspace -u mpi \
    --entrypoint bash mpi-lab:cuda -lc '
        cmake -S /workspace -B /tmp/build -DCMAKE_BUILD_TYPE=Release
        cmake --build /tmp/build --target msearch --parallel
        /tmp/build/bin/msearch --list-backends'
```

```console
BACKEND   STATUS       DESCRIPTION
cuda      available    NVIDIA GPU (CUDA, persistent buffers, one thread per placement)
openmp    available    multicore CPU (OpenMP, parallel over placements)
serial    available    single-threaded reference implementation
```

Cross-check the GPU against the serial reference before trusting any GPU result
or timing:

```console
$ msearch --verify -i examples/hybrid-object-matching/tests/data/reference.txt
reference: serial (10 pictures)
[info ] cuda backend: device 0 (NVIDIA GeForce RTX 3050 4GB Laptop GPU, 20 SMs)
  cuda     OK (identical to serial)
  openmp   OK (identical to serial)
```

The pre-built artifact image is simpler still if you only want to run the
object-matching example rather than develop it:

```bash
docker build -f Dockerfile.cuda -t msearch:cuda .
docker run --rm --gpus all msearch:cuda msearch --list-backends
docker run --rm --gpus all msearch:cuda msearch --verify -i tests/data/reference.txt
```

## GPU nodes in the simulated cluster

```bash
./scripts/start-cluster.sh --gpu 2
./scripts/run-mpi.sh -n 2 msearch --backend cuda \
    -i examples/hybrid-object-matching/tests/data/reference.txt -o -
```

This works, and the result is byte-identical to the serial reference. It is
useful for one specific question: *does my CUDA kernel still produce the right
answer when several MPI ranks are driving it?*

**It is not useful for anything about performance.** Every node in the overlay
maps onto the **same physical card**. Two containers do not become two GPUs;
they become two processes contending for one device's SMs and memory:

- two ranks sharing one GPU are usually **slower** than one rank owning it —
  the opposite of what the same job does on a real two-GPU cluster
- memory is the binding constraint. A 4 GB card split between four ranks is
  1 GB each, and the CUDA backend will report an allocation failure long before
  the CPU nodes would
- the GPU overlay therefore defaults to **2 nodes**, not 4

If you want GPU numbers, run one node and read
[the object-matching example's PERFORMANCE.md](../examples/hybrid-object-matching/docs/PERFORMANCE.md),
which documents how its figures were measured and what they do and do not
support.

## Building without a GPU

`docker build` cannot access a GPU — BuildKit does not honour `--gpus` — so the
CUDA image builds fine on a machine that has none, and CI does exactly that.
That is not a workaround; it exercises the graceful-degradation path:

- with no device visible, the CUDA backend reports itself **unavailable with a
  reason**, and `--backend auto` falls through to OpenMP
- the test suite inside the build runs on the CPU backends
- forgetting `--gpus all` at run time therefore produces a correct, slower run
  rather than a crash

```console
$ docker run --rm msearch:cuda msearch --list-backends
BACKEND   STATUS       DESCRIPTION
cuda      unavailable  NVIDIA GPU (CUDA, persistent buffers, one thread per placement)
                       (no CUDA-capable device detected)
openmp    available    multicore CPU (OpenMP, parallel over placements)
serial    available    single-threaded reference implementation
```

## Version choices

**CUDA 12.4 on Ubuntu 22.04**, chosen for driver reach rather than novelty.
CUDA has minor-version compatibility across a major release, so a binary built
against 12.4 runs on any driver supporting 12.x — in practice NVIDIA Windows
driver 527.41 (Nov 2022) or newer. Pinning a newer minor would raise the floor
for no benefit: nothing here uses a feature added after 12.0.

Override if you need to:

```bash
docker build -f Dockerfile.cuda --build-arg CUDA_VERSION=12.6.1 -t mpi-lab:cuda .
```

**GPU architectures** are Pascal through Hopper as real code, plus PTX from
`sm_90` so newer GPUs still work by JIT. Cutting this to one target makes the
build much faster:

```bash
docker build -f Dockerfile.cuda --build-arg CUDA_ARCHITECTURES="86-real" \
    --target dev -t mpi-lab:cuda .
```

## Writing a CUDA example

```cmake
mpilab_add_example(my-gpu-example
  SOURCES main.c kernel.cu
  MPI
  CUDA
  DESCRIPTION "what it shows")
```

`CUDA` is a hard requirement: the example is skipped with a status line where
`nvcc` is absent, so the same `cmake` command still works on the CPU nodes.

Two things the object-matching example gets right and are worth copying:

**Probe at run time, not compile time.** Its `available()` asks the driver
whether a device is actually visible, so one binary can be built with CUDA
compiled in and still run on a machine without a card. Compiling the kernel and
finding the GPU are different questions.

**Map ranks to devices by node-local rank.** Several ranks on one node must not
all grab device 0. It splits `MPI_COMM_WORLD` by
`MPI_COMM_TYPE_SHARED` to find its position within its own node, and indexes
the device list with that — one line, and it is what makes
`--ntasks-per-node=4 --gres=gpu:4` work on a real cluster with no code change.
In this container overlay there is only one device, so every rank shares it;
the mapping logic is unchanged and simply has nowhere to spread to.
