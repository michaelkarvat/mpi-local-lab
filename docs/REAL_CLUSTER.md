# Moving to a real cluster

Nothing in this repository is container-specific. The same source, the same
`CMakeLists.txt` and the same binaries run on real hardware — the launcher
changes and the code does not.

This document covers what transfers, what does not, and which assumptions the
local environment quietly lets you get away with.

---

## The short version

```bash
git clone <this repo>
cd mpi-local-lab

scripts/slurm/build.sh                  # build once, on the login node
sbatch scripts/slurm/submit.sbatch
```

`scripts/slurm/build.sh` loads the usual modules (adjust to your site's names),
configures and builds. `submit.sbatch` requests nodes and runs the binary under
`srun`.

Building on the login node rather than inside the job is deliberate: a compile
error surfaces in seconds instead of consuming an allocation, and repeated
submissions do not recompile identical sources.

## What transfers unchanged

**The programs.** No example contains a line of Docker-aware code. Rank ids,
collectives, decompositions and the hybrid MPI+OpenMP structure are all
standard MPI.

**The build.** The root `CMakeLists.txt` detects MPI, OpenMP and CUDA rather
than requiring them, so the same configure command works on a login node with
modules loaded, in a container, and on a bare laptop. `cmake -S . -B build`
either way.

**The tests.** `ctest` runs on a compute node exactly as it does in a
container. The rank sweeps are more meaningful there, not less — they are now
crossing a real network.

**The correctness guarantees.** If `distributed-sum` gives the same answer at
1, 2 and 4 ranks locally, it will on the cluster. That is what those tests are
for, and it is the main thing the local environment buys you.

## What changes

| | Local containers | Real cluster |
|---|---|---|
| Launch | `mpirun` over ssh, hostfile from Docker DNS | `srun`, or `mpirun` under the scheduler |
| Node list | `getent ahostsv4 node` | `$SLURM_JOB_NODELIST` |
| Cores per rank | every core the host has, times the node count | what the scheduler reserved: `$SLURM_CPUS_PER_TASK` |
| Filesystem | one bind mount every node shares | a shared filesystem, usually slower, sometimes not shared |
| GPUs | one physical card, shared by every node | one or more per node, via `--gres` |
| Network | a virtual bridge on one host | InfiniBand or Ethernet, with real latency |

### Do not pass a hostfile

`scripts/run-mpi.sh` builds one because there is no scheduler here to ask. On a
cluster the scheduler already knows the node list, and `srun` — or `mpirun`
launched inside an allocation — picks it up automatically. Passing your own
hostfile there is a way to fight the scheduler and lose.

### Set thread counts from the allocation

The local environment lets a rank see every core on the host, and with four
nodes on one machine that is every core four times over. It mostly does not
matter because nothing here is a performance measurement.

On a cluster it matters immediately. `submit.sbatch` does the right thing:

```bash
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OMP_PROC_BIND=close
export OMP_PLACES=cores
```

Without that, each rank spawns a thread per core on the node and the ranks
sharing that node oversubscribe it several times over. The object-matching
example measured **14× slower** from exactly this mistake.

### GPUs actually multiply

```bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
```

One rank per node, one GPU per rank. Raising `--ntasks-per-node` to match
`--gres=gpu:N` fans the ranks across a node's GPUs with no code change, because
the CUDA backend maps each rank onto a device by its rank *within its node*.
That logic is already exercised locally — it just has only one device to choose
from there.

## Assumptions the local environment lets you get away with

**A shared filesystem.** Here, `/workspace` is one bind mount every node sees,
so a file written by rank 0 is instantly visible to rank 3. Many clusters have
a shared filesystem too, but it is a network filesystem: slower, with different
consistency, and occasionally node-local scratch instead. Code that has every
rank read the same file at once will be fine here and a thundering herd there.

The object-matching example already assumes rank 0 owns the filesystem — one
process reads the input, one writes the output — which is the pattern that
survives the move.

**Latency you can ignore.** A round trip between containers is microseconds
over a loopback bridge. On a real interconnect it is larger and, more
importantly, *variable*. An algorithm that sends many small messages looks fine
here and does not there. If your program's structure depends on communication
being cheap, that assumption is untested until it runs on hardware.

**Homogeneous nodes.** Every container is the same image on the same CPU. Real
clusters have mixed generations, partitions with different core counts, and
nodes that are busier than others. Static work partitioning that looks balanced
here can leave ranks idle there — which is why the object-matching example
claims work dynamically instead.

**Unlimited time.** No wall clock here. Real jobs are killed at the limit, so
long runs need checkpointing or a smaller problem.

**Root-ish freedom.** You can `apt install` inside a node. On a login node you
cannot; you get modules, or you build dependencies yourself.

## Measuring performance

Do it on the cluster, not here. See
[ARCHITECTURE.md § What is not simulated](ARCHITECTURE.md#what-is-not-simulated)
for why local numbers cannot support a multi-node claim.

When you do measure, the object-matching example's
[PERFORMANCE.md](../examples/hybrid-object-matching/docs/PERFORMANCE.md) is a
worked example of doing it defensibly: a serial baseline so speedups have a
denominator, best-of-N rather than a single run, and explicit statements about
what varied between measurements. Its most useful lesson is negative — a GPU
kernel that was correct and 3.7× *slower* than one CPU core, found only because
there was a baseline to compare against.

## A checklist

Before your first real submission:

- [ ] `ctest` passes locally, including the rank sweeps
- [ ] `bash tests/cluster/run_all.sh` passes — collectives work over a network
- [ ] Your program does not assume a rank count, or checks the one it got
- [ ] Output is reassembled in input order, not arrival order
- [ ] Thread count comes from the environment, not a hardcoded number
- [ ] Only rank 0 writes the output file
- [ ] Failures propagate through the collectives — a rank that exits early
      leaves the others blocked forever
- [ ] `scripts/slurm/build.sh` module names match your site
- [ ] Wall clock, node count and `--gres` in `submit.sbatch` match what you
      actually need
