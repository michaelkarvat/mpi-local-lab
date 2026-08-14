# Architecture

How a handful of containers becomes something MPI treats as a cluster, and
which parts of a cluster that deliberately does not reproduce.

For the architecture of the object-matching workload — backends, the
determinism contract, the CUDA kernel design — see
[examples/hybrid-object-matching/docs/ARCHITECTURE.md](../examples/hybrid-object-matching/docs/ARCHITECTURE.md).
This document is about the environment.

---

## The one decision everything else follows from

Two questions are independent, and keeping them independent is the design:

| | |
|---|---|
| **Where do processes run?** | the **runtime** — one process · local ranks · this container cluster · a real cluster |
| **How does one rank compute?** | the **backend** — serial · OpenMP · CUDA |

An example never learns which runtime launched it, and the runtime never learns
what the example does inside a rank. The consequence is that
`MPI + Serial`, `MPI + OpenMP`, `MPI + CUDA` and `MPI + OpenMP + CUDA` are
configurations rather than variants, and moving a program from your laptop to
these containers to a real SLURM allocation changes the launcher and nothing
else.

The container cluster is one point on the runtime axis. It is not a new
programming model, and no example contains a line of Docker-aware code.

---

## What a node is

```text
                        Host Machine
                             │
                       Docker Engine
                             │
                   mpi-lab-net (bridge, private)
                             │
     ┌───────────────┬───────┴───────┬───────────────┐
     │               │               │               │
  node-1          node-2          node-3          node-4
  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
  │ sshd     │    │ sshd     │    │ sshd     │    │ sshd     │
  │ mpirun   │    │ orted    │    │ orted    │    │ orted    │
  │ gcc,cmake│    │          │    │          │    │          │
  │ OpenMPI  │    │ OpenMPI  │    │ OpenMPI  │    │ OpenMPI  │
  └──────────┘    └──────────┘    └──────────┘    └──────────┘
     │  │            │  │            │  │            │  │
     │  └────────────┴──┴────────────┴──┴────────────┘  │
     │            /build  (named volume, shared)         │
     └───────────────────────────────────────────────────┘
                  /workspace  (bind mount, your tree)
```

Every node is the same image. There is no head node in the image — node 1 is
where the scripts happen to run `mpirun`, and any node would do.

### Why ssh

OpenMPI starts remote processes with a *process launch module*. On a cluster
that is usually SLURM's; without a scheduler it is `rsh/ssh`. There is no
Docker launcher, and writing one would mean reimplementing the part of MPI this
repository is trying to demonstrate honestly.

So each node runs `sshd`, and `mpirun` on node 1 ssh's to the others exactly as
it would on real hardware. That is a feature: the launch path you debug here is
the launch path you will debug on a cluster.

### Why a key is baked into the image

The setup step this project exists to remove is "generate a key and distribute
it to every node". Since every node runs the *same image*, a key pair generated
once at build time is already present and already authorised everywhere, and
the cluster works on first boot with nothing for the user to do.

The alternatives were considered and rejected:

| Approach | Why not |
|---|---|
| Generate keys on first container start into a shared volume | Races between replicas starting simultaneously; needs a lock, and a failed lock is a cluster that half-works |
| Mount the user's own `~/.ssh` | Puts the user's real private key inside a container running a network service, to solve a problem that does not need it |
| Generate on the host in `start-cluster.sh` | Adds `ssh-keygen` to the host prerequisites and a setup step to the quick start |
| An MPI launcher over `docker exec` | Would have to reimplement OpenMPI's PLM, and the resulting launch path would look nothing like a real cluster's |

The cost is bounded by how these containers are used: an internal bridge
network, **no published ports**, and nothing inside but example source. The
image is not something to push to a registry, and `docker/setup-node.sh` says
so at the point the key is created.

### Why one replicated service, not `node1`…`node4`

Writing four services out gives readable hostnames. It also means changing the
node count is a YAML edit, and "change the number of nodes" is exactly the
thing this environment should make trivial. One replicated service makes the
count a command-line argument:

```bash
./scripts/start-cluster.sh 6
```

The price is that Docker Compose cannot template a hostname per replica, so
each container's hostname is its container id — and that id is what
`MPI_Get_processor_name` reports. `./scripts/status.sh` prints the mapping, and
`run-mpi.sh` prints a one-line legend before each launch, which is what makes
`hello-mpi`'s output readable.

### Why the source is bind-mounted and the build output is not

```yaml
volumes:
  - .:/workspace      # bind mount
  - build:/build      # named volume
```

The source is a bind mount because the point of a *development* environment is
that you edit with your own tools on the host and the change is visible
immediately, with no image rebuild.

Build output is a named volume for two reasons. It is shared, so node 1
compiles once and every rank launches the same file from the same path — which
is what `mpirun` requires and what would otherwise need a copy step. And on
Windows and macOS a bind mount crosses a filesystem boundary where writing
thousands of object files is slow; a named volume lives inside the Docker VM
and is not. It also keeps a native host build in `./build` from fighting with
the container's CMake cache.

### Why the transports are pinned

`docker/openmpi-mca-params.conf` pins both the data path and the out-of-band
channel to `eth0`:

```conf
btl_tcp_if_include = eth0
oob_tcp_if_include = eth0
```

A container on a compose network is multi-homed. OpenMPI probes every
interface and can select one the peer cannot route to, and when it does,
`mpirun` **does not fail — it blocks**, with no error, until killed. It is the
most common way a containerised MPI setup appears broken, and it presents as a
bug in the user's program. Pinning removes the guesswork.

---

## How a run happens

`./scripts/run-mpi.sh -n 4 examples/hello-mpi`:

1. **Resolve.** `examples/hello-mpi` → the CMake target `hello-mpi`. The
   mapping is written to `/build/examples.map` at configure time rather than
   guessed, because `examples/hybrid-object-matching` builds a binary called
   `msearch` and no naming convention covers that.
2. **Build.** Node 1 configures `/workspace` into `/build` if it has not
   already, then builds that one target. Incremental after the first time.
3. **Discover.** Node 1 asks Docker's embedded DNS what `node` resolves to. The
   compose service name returns every replica's address, so the cluster
   describes itself and nothing on the host has to inspect containers. The
   result becomes `/build/hostfile`.
4. **Launch.** `mpirun -np 4 --hostfile /build/hostfile --map-by node`. Ranks
   are placed round-robin across nodes, so 4 ranks on 4 nodes is one each. The
   alternative — filling each node's slots before moving on — would put all
   four ranks of a small run on node 1 and stop exercising the network at all.

The hostfile is regenerated every run. The cluster may have been rescaled since
the last one, and a stale hostfile naming a departed node hangs the launch.

---

## What is *not* simulated

This is the section to read before drawing a conclusion from anything measured
here.

The containers are **logical nodes**. They share:

- **the same CPU** — four "nodes" on a 4-core laptop are four processes
  competing for four cores, not four machines
- **the same memory and memory bandwidth**
- **the same network adapter** — traffic between nodes is a virtual bridge on
  the host and never touches a wire
- **the same disk**
- **the same GPU**, under the GPU overlay

What follows:

| Question | Answerable here? |
|---|---|
| Does my collective deadlock at 3 ranks? | **yes** — and it will deadlock the same way on a cluster |
| Does my program give the same answer at 1, 2 and 4 ranks? | **yes** — this is what the rank sweeps check |
| Does my launch configuration work across hosts? | **yes** — ssh, hostfiles and DNS are all real here |
| Is my domain decomposition correct at an uneven split? | **yes** |
| How much faster is 4 nodes than 1? | **no** — there is no extra hardware |
| What is my interconnect latency? | **no** — you are timing a loopback bridge |
| Will this scale to 64 nodes? | **no** |

`examples/ping-pong` prints a round-trip time on purpose, and says in its own
source that the number is not a network measurement. Timing something and
knowing what the timing means are different skills, and the example is a good
place to practise the second.

Performance claims involving multiple physical nodes have to come from a real
cluster. See [REAL_CLUSTER.md](REAL_CLUSTER.md).

---

## Images and stages

```text
Dockerfile
  builder  →  compiles the whole tree, runs ctest. Never shipped.
  dev      →  a node: gcc, cmake, OpenMPI, sshd, gdb. No source baked in.
  runtime  →  the object-matching binary and its runtime libraries. No compiler.

Dockerfile.cuda
  the same three, on nvidia/cuda, with nvcc and the CUDA runtime.
```

`builder` runs the full test suite during `docker build`, so a broken commit
cannot produce a usable `runtime` image. That guarantee does **not** extend to
`dev`, which by design carries no source to test — which is why CI runs `ctest`
explicitly inside a live cluster as well.

The default target is `runtime`, so a bare `docker build .` still produces the
artifact image the object-matching example documents.

`docker/setup-node.sh` is shared by both `dev` stages rather than duplicated.
The CPU and GPU nodes must agree exactly; a difference in their ssh setup would
present as a cluster that works until you switch to the GPU overlay.

---

## Testing strategy

Two suites, because there are two different things to be wrong.

**CTest, inside one container** — unit tests, golden output, and a rank sweep
per example. These would pass on a machine with no Docker at all. They check
the programs.

**`tests/cluster/run_all.sh`, from the host against live containers** — DNS
resolution between nodes, ssh reachability, ranks landing on distinct
hostnames, rank-count invariance over the real network, and the object-matching
golden file reproduced across containers. These check the environment, and
nothing in the first suite would notice if the second broke.

The rank sweep deserves a note. `cmake/RunRanks.cmake` runs an example at
several rank counts and requires byte-identical output, which is the property
most MPI bugs break first — a missing `Gatherv` displacement, an off-by-one in
a decomposition, an accumulation whose order depends on arrival. It is
generalised from the object-matching example's `mpi_equivalence` test, which
proved the same thing for one program before there was an environment to prove
it for all of them.
