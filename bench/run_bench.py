#!/usr/bin/env python3
"""Benchmark harness: measures backends against the serial baseline.

Produces the markdown table in docs/PERFORMANCE.md. Every speedup figure in
this repository comes from this script rather than from an estimate -- which is
the reason the serial backend exists at all.

    bench/run_bench.py --binary build/msearch
    bench/run_bench.py --binary build/msearch --threads 1 2 4 8 --reps 5
    bench/run_bench.py --binary build/msearch --input my.txt --backends serial openmp cuda
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from typing import Dict, List, Optional

# Matches the --bench line emitted by src/cli/main.c.
BENCH_LINE = re.compile(
    r"backend=(?P<backend>\S+)\s+ranks=(?P<ranks>\d+)\s+reps=(?P<reps>\d+)\s+"
    r"min=(?P<min>[\d.]+)s\s+median=(?P<median>[\d.]+)s\s+mean=(?P<mean>[\d.]+)s\s+"
    r"max=(?P<max>[\d.]+)s"
)

DEFAULT_WORKLOAD = dict(pictures=4, picture_size=512, objects=4, object_size=16)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", default="build/msearch", help="path to the msearch binary")
    parser.add_argument("--input", help="problem file (generated if omitted)")
    parser.add_argument("--backends", nargs="+", default=["serial", "openmp", "cuda"])
    parser.add_argument("--threads", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--ranks", nargs="+", type=int, default=[],
                        help="also run under mpirun with these rank counts")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2024)
    return parser.parse_args(argv)


def available_backends(binary: str) -> List[str]:
    out = subprocess.run([binary, "--list-backends"], capture_output=True, text=True, check=True)
    return [
        line.split()[0]
        for line in out.stdout.splitlines()[1:]
        if len(line.split()) >= 2 and line.split()[1] == "available"
    ]


def generate_workload(seed: int) -> str:
    """A worst-case input: nothing is planted, so every placement is scanned."""
    os.makedirs("bench/data", exist_ok=True)
    path = "bench/data/workload.txt"
    if os.path.exists(path):
        return path
    generator = os.path.join(os.path.dirname(__file__), "..", "tools", "gen_input.py")
    subprocess.run(
        [sys.executable, generator,
         "--pictures", str(DEFAULT_WORKLOAD["pictures"]),
         "--picture-size", str(DEFAULT_WORKLOAD["picture_size"]),
         "--objects", str(DEFAULT_WORKLOAD["objects"]),
         "--object-size", str(DEFAULT_WORKLOAD["object_size"]),
         "--threshold", "1.0", "--plant-rate", "0", "--seed", str(seed), "-o", path],
        check=True,
    )
    return path


def run(binary: str, path: str, backend: str, threads: Optional[int], reps: int,
        ranks: int) -> Optional[Dict[str, float]]:
    command: List[str] = []
    if ranks > 1:
        launcher = shutil.which("mpirun") or shutil.which("mpiexec")
        if launcher is None:
            return None
        command += [launcher, "-n", str(ranks), "--oversubscribe"]
    command += [binary, "--backend", backend, "--bench", str(reps), "-i", path, "-o", os.devnull,
                "--quiet"]
    if threads:
        command += ["--threads", str(threads)]

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ! {backend} threads={threads} ranks={ranks} failed:\n{result.stderr}",
              file=sys.stderr)
        return None
    # Timing goes to stderr so that `--output -` keeps stdout clean for results;
    # both streams are scanned so the harness does not care which it lands on.
    match = BENCH_LINE.search(result.stderr) or BENCH_LINE.search(result.stdout)
    if match is None:
        return None
    return {key: float(match.group(key)) for key in ("min", "median", "mean", "max")}


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    if not os.path.exists(args.binary):
        print(f"binary not found: {args.binary}", file=sys.stderr)
        return 1
    # Absolute, so the launcher does not have to resolve a relative path (which
    # Windows' CreateProcess declines to do for paths containing a separator).
    args.binary = os.path.abspath(args.binary)

    usable = available_backends(args.binary)
    path = args.input or generate_workload(args.seed)
    print(f"input:    {path}")
    print(f"backends: {', '.join(usable)}\n")

    baseline: Optional[float] = None
    rows: List[tuple] = []

    for backend in args.backends:
        if backend not in usable:
            print(f"skipping {backend} (not available)")
            continue
        thread_counts = args.threads if backend == "openmp" else [None]
        for threads in thread_counts:
            for ranks in [1] + [r for r in args.ranks if r > 1]:
                stats = run(args.binary, path, backend, threads, args.reps, ranks)
                if stats is None:
                    continue
                # The serial, single-rank run is the reference for every speedup.
                if baseline is None and backend == "serial" and ranks == 1:
                    baseline = stats["min"]
                speedup = (baseline / stats["min"]) if baseline else float("nan")
                rows.append((backend, threads or "-", ranks, stats["min"], stats["median"],
                             speedup))
                print(f"  {backend:<7} threads={str(threads or '-'):<3} ranks={ranks} "
                      f"min={stats['min']:.4f}s  speedup={speedup:.2f}x")

    print("\n| Backend | Threads | Ranks | Best (s) | Median (s) | Speedup |")
    print("|---------|---------|-------|----------|------------|---------|")
    for backend, threads, ranks, best, median, speedup in rows:
        print(f"| {backend} | {threads} | {ranks} | {best:.4f} | {median:.4f} | {speedup:.2f}x |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
