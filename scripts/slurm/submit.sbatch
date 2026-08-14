#!/bin/bash
#SBATCH --job-name=msearch
#SBATCH --output=logs/msearch-%j.out
#SBATCH --error=logs/msearch-%j.err
#SBATCH --time=0-00:30:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#
# One rank per node, one GPU per rank. The CUDA backend maps each rank onto a
# device by its rank *within the node*, so raising --ntasks-per-node to match
# --gres=gpu:N fans the ranks out across a node's GPUs with no code change.
#
# Submit with:  sbatch scripts/slurm/submit.sbatch [extra msearch flags]
#
# Note this script only *runs* the program. The previous version recompiled
# every source file on every submission, inside the job's wall-clock budget and
# with no way to notice a build failure before the allocation was already
# burning. Build once on the login node with scripts/slurm/build.sh.

set -euo pipefail

INPUT=${MSEARCH_INPUT:-input.txt}
OUTPUT=${MSEARCH_OUTPUT:-output.txt}
BACKEND=${MSEARCH_BACKEND:-auto}
BINARY=${MSEARCH_BINARY:-build/bin/msearch}

mkdir -p logs

if [[ ! -x "$BINARY" ]]; then
    echo "error: $BINARY not found. Run scripts/slurm/build.sh first." >&2
    exit 1
fi

# Give each rank the cores SLURM actually reserved for it, instead of letting
# OpenMP see every core on the node and oversubscribe.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "nodes=${SLURM_JOB_NUM_NODES:-?} tasks=${SLURM_NTASKS:-?} cpus/task=${OMP_NUM_THREADS}"
"$BINARY" --version

srun "$BINARY" --backend "$BACKEND" --input "$INPUT" --output "$OUTPUT" "$@"
