# Performance

Every number here was produced by `bench/run_bench.py` on the machine described
below. Where a figure has **not** been measured, it says so — this document
contains no estimates presented as results.

## Method

```bash
# 58 MB worst case: nothing is planted, so every placement is scanned
python tools/gen_input.py --pictures 4 --picture-size 2048 --objects 4 \
                         --object-size 16 --threshold 1.0 --plant-rate 0 \
                         --seed 2024 -o bench/data/large.txt

python bench/run_bench.py --binary build/msearch --input bench/data/large.txt \
                         --reps 3 --threads 1 2 4 8 16
```

The workload plants nothing deliberately: with a match present the search stops
early and the measurement becomes a measurement of *where the match was*. A
full scan is the reproducible case.

The reported figure is the **minimum** of the repetitions — the run least
polluted by scheduler noise, and the right basis for a ratio. Medians are
printed alongside so a suspiciously fast outlier is visible.

**Hardware.** Intel Core i7-1360P (4 performance cores + 8 efficiency cores,
16 hardware threads), Windows 11, GCC 14.2, `-O3`. This is a laptop, not a
compute node; the CPU scaling ceiling below reflects that.

**Workload.** 4 pictures of 2048×2048, 4 objects of 16×16, threshold 1.0
→ 4 × 4 × (2048−16+1)² ≈ **66.1 M placements**.

## Early exit

The largest single win, and it costs three lines in `include/msearch/metric.h`.
Every term of the score is non-negative, so the partial sum only increases;
once it reaches the threshold the placement is abandoned.

| Serial backend | Best of 3 | Relative |
|---|---|---|
| **with** early exit | **0.997 s** | **1.00×** |
| without early exit | 16.890 s | 16.95× slower |

Output is **byte-identical** with and without it, verified on all five fixtures
and on the 58 MB benchmark input. It is a pure optimisation, and because it
lives in the shared metric every backend inherits it.

At 66.1 M placements in 0.997 s that is **≈ 66 M placements/s** single-threaded,
or ≥ 1.06 G element evaluations/s (each placement evaluates at least the
object's first row of 16 elements, each a subtract, divide, absolute value and
add).

## CPU scaling

| Backend | Threads | Best (s) | Median (s) | Speedup |
|---------|---------|----------|------------|---------|
| serial  | –  | 0.9693 | 0.9740 | 1.00× |
| openmp  | 1  | 1.0150 | 1.0287 | 0.95× |
| openmp  | 2  | 0.5763 | 0.5844 | 1.68× |
| openmp  | 4  | 0.3373 | 0.3432 | 2.87× |
| openmp  | 8  | 0.2532 | 0.2618 | 3.83× |
| openmp  | 16 | 0.2102 | 0.2206 | 4.61× |

**Reading this honestly:**

- **1 thread is 5% slower than serial.** That is the cost of the parallel
  region and the atomic read of the running minimum. Reporting it is more
  useful than hiding it — it is the correct denominator for "what did
  parallelism actually cost me".
- **2 and 4 threads scale at 84% and 72% efficiency** on the performance cores.
- **Beyond 4 threads efficiency falls to ~29%.** Two causes, both real: this
  CPU has only 4 performance cores, so threads 5–12 land on efficiency cores
  that are roughly half as fast, and threads 13–16 are SMT siblings sharing
  execution units. On a homogeneous compute node (the SLURM script requests
  8 cores per task) the curve should stay closer to linear — **not measured
  here**, because the cluster is not available for this write-up.
- The inner loop after early exit touches one object row and one picture row
  per placement, with little reuse, so the kernel is closer to
  memory-bandwidth-bound than compute-bound. That caps scaling independently of
  core count.

## GPU

**Not measured.** No CUDA device was available while this was written. The CUDA
backend compiles and its structure is described in
[ARCHITECTURE.md](ARCHITECTURE.md#cuda-design), but no throughput or speedup
figure is claimed for it, and none should be inferred from this document.

To fill this section in on a machine with a GPU:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel
./build/msearch --verify -i tests/data/reference.txt      # correctness first
python bench/run_bench.py --binary build/msearch --input bench/data/large.txt \
                         --backends serial openmp cuda
```

`--verify` must pass before any timing is meaningful: it checks the GPU results
against the serial reference for exact equality.

## Reproducing

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel
ctest --test-dir build --output-on-failure
make bench          # or: python bench/run_bench.py --binary build/msearch
```

`tools/gen_input.py` is seeded, so the benchmark input is byte-reproducible
from the command at the top of this file.
