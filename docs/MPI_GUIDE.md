# Working in the lab

Day-to-day use: writing an MPI program, running it, debugging it when it
misbehaves, and the container-specific things that will trip you up.

---

## The cycle

```bash
./scripts/start-cluster.sh            # once
# ... edit examples/my-example/main.c in your editor, on the host ...
./scripts/run-mpi.sh -n 4 my-example  # compiles and runs, a second or two
```

There is no image rebuild in that loop. Your working tree is bind-mounted at
`/workspace` in every node, so an edit on the host is visible to all four
immediately; `run-mpi.sh` rebuilds only what changed, into the shared `/build`
volume that every node mounts.

You only need to rebuild the image if you change the *image* — the Dockerfile,
or a package a node needs.

## Adding an example

```text
examples/my-example/
  ├── CMakeLists.txt
  └── main.c
```

```cmake
mpilab_add_example(my-example
  SOURCES main.c
  MPI
  DESCRIPTION "one line, shown in the examples table")

mpilab_add_rank_test(my-example RANKS 1 2 4 MODE invariant)
```

Then add it to the list in the root `CMakeLists.txt`:

```cmake
add_subdirectory(examples/my-example)
```

`MPI` is a hard requirement — without it the example is skipped with a status
line rather than failing the configure, which is what lets one `cmake` command
work on a bare laptop and inside a GPU node alike. `OPENMP` is opportunistic:
list it, guard your pragmas with `#ifdef _OPENMP`, and the same source compiles
single-threaded where OpenMP is absent.

### The rank test is the point

```cmake
mpilab_add_rank_test(my-example RANKS 1 2 3 4 MODE invariant)
```

Runs the example at each rank count and requires **byte-identical output**.
This catches most MPI bugs before they become mysterious: a wrong `Gatherv`
displacement, a decomposition that assumes even division, a result assembled in
arrival order instead of input order.

Two modes:

| `MODE` | Requires | Use for |
|---|---|---|
| `invariant` | identical output at every rank count | anything with a defined answer |
| `ranks-present` | at N ranks, N lines, each naming a distinct rank | programs whose lines legitimately interleave |

Include a rank count that does **not** divide your problem size — `3` for a
size of 100. Even division is the case that hides the bug.

### Keep nondeterministic output off stdout

Timings, rank counts and thread counts change between runs, so they belong on
stderr. The examples all do this:

```c
printf("sum = %lld\n", total);                       /* result   -> stdout */
fprintf(stderr, "ranks=%d elapsed=%.6fs\n", n, dt);  /* diagnostic -> stderr */
```

That is what makes `MODE invariant` possible, and it also means
`run-mpi.sh ... > results.txt` captures results and nothing else.

### Floating point is not associative

`MPI_Reduce` with `MPI_SUM` over doubles groups the additions differently at 2
ranks than at 4, and the low bits of the answer differ. The program looks
correct and the rank sweep fails intermittently.

`examples/distributed-sum` reduces **integers** for exactly this reason, and
says so in its header. If your answer is floating point, either compare with a
tolerance or accept that the rank count is observable — but decide, rather than
discovering it from a flaky test.

## Running

```bash
./scripts/run-mpi.sh -n 4 examples/hello-mpi     # path form, as in the docs
./scripts/run-mpi.sh -n 4 hello-mpi              # or just the name
./scripts/run-mpi.sh -n 8 distributed-sum 5000000
./scripts/run-mpi.sh -n 4 --no-build matrix-multiply 512
```

`-n` is **ranks**. Node count is however many containers are running. Ranks go
round-robin over nodes, so on a 4-node cluster `-n 4` is one rank per node and
`-n 8` is two.

`--no-build` skips the compiler when you know nothing changed. It does not skip
the name lookup, and it will tell you plainly if the binary was never built.

### Hybrid MPI + OpenMP

Threads come from the environment, and the default is every core the container
can see — which, with four nodes on one host, is every core four times over:

```bash
docker compose exec -e OMP_NUM_THREADS=2 --index=1 -u mpi node \
    mpirun -np 4 --hostfile /build/hostfile --map-by node /build/bin/matrix-multiply
```

On a real cluster the scheduler tells you how many cores you actually own; see
`scripts/slurm/submit.sbatch`, which sets `OMP_NUM_THREADS` from
`SLURM_CPUS_PER_TASK`.

## Debugging

```bash
./scripts/shell.sh        # a shell on node 1, as the mpi user
./scripts/shell.sh 3      # node 3
```

Inside a node you have the full toolchain, `mpirun`, `gdb`, `ip`, `ping` and
`ssh`. Useful things to do there:

```bash
# Is the cluster visible from in here?
getent ahostsv4 node

# Can I reach a peer the way mpirun will?
ssh 172.18.0.3 hostname

# Run a rank under gdb (one rank only, or the terminals fight)
mpirun -np 1 gdb --args /build/bin/distributed-sum 1000

# Which rank is on which node, with output tagged
mpirun -np 4 --hostfile /build/hostfile --map-by node --tag-output \
    /build/bin/hello-mpi
```

`MPI_Barrier` before a crash and `fflush(stdout)` after a `printf` are worth
more than they look: without the flush, output from a rank that aborts is lost,
and the last thing you see is from a different rank.

## Things that will trip you up

**`mpirun` hangs with no output.** Almost always interface selection —
OpenMPI picked a route the peer cannot reach. `docker/openmpi-mca-params.conf`
pins `eth0` to prevent this. If you add a second network to `compose.yaml`,
revisit that file.

**"Permission denied (publickey)".** sshd is refusing the login. Check
`docker compose logs node`; if it says *account is locked*, something changed
`UsePAM` — see the note in `docker/setup-node.sh`.

**The binary is missing on other nodes.** It should be in `/build`, which every
node mounts. If you built somewhere else, only node 1 can see it.

**Your edit had no effect.** Check you edited under the repository, not inside a
container's `/build`. `/workspace` is your tree; `/build` is generated.

**Scripts fail oddly on Windows.** Run them from Git Bash or WSL, not
PowerShell or cmd. `scripts/lib.sh` sets `MSYS_NO_PATHCONV` and
`MSYS2_ARG_CONV_EXCL` because MSYS otherwise rewrites `/build/bin/hello-mpi`
into a Windows path before Docker ever sees it.

**Compiling is slow.** On Windows, cloning into the WSL2 filesystem
(`\\wsl$\Ubuntu\home\you\...`) rather than `C:\` avoids the bind mount crossing
a filesystem boundary and is markedly faster.

**A stale CMake cache.** `./scripts/stop-cluster.sh --purge` deletes the build
volume; the next start configures from scratch.

## Testing your work

```bash
# Programs, inside one node
./scripts/shell.sh
  cmake -S /workspace -B /build && ctest --test-dir /build --output-on-failure

# The environment, from the host
bash tests/cluster/run_all.sh
```

Both run in CI. The second is the one that would notice if the cluster stopped
being a cluster.
