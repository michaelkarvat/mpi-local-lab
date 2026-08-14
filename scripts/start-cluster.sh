#!/usr/bin/env bash
# Bring the cluster up and wait until every node is actually ready.
#
#   ./scripts/start-cluster.sh          # four nodes, the default
#   ./scripts/start-cluster.sh 8        # eight
#   ./scripts/start-cluster.sh --gpu 2  # CUDA nodes -- read compose.gpu.yaml first
#
# `docker compose up -d` on its own is enough to create the containers, and the
# PRD quick start uses exactly that. This adds the part that matters when the
# next command is a script rather than a human: `up -d` returns as soon as the
# containers are *created*, which is before sshd is listening. Launching mpirun
# in that window fails with a connection error that looks like a configuration
# problem. Waiting on the health check removes the race.

set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

DEFAULT_NODES=4
while [ $# -gt 0 ]; do
    case "$1" in
        # GPU nodes default to 2, not 4: they all share one physical device and
        # its memory divides between them. See the header of compose.gpu.yaml.
        --gpu) export MPILAB_GPU=1; DEFAULT_NODES=2; shift ;;
        -h|--help) sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        -*) die "unknown option '$1' (try --help)" ;;
        *) break ;;
    esac
done

NODES=${1:-$DEFAULT_NODES}
[[ "$NODES" =~ ^[0-9]+$ ]] && [ "$NODES" -ge 1 ] || die "node count must be a positive integer, got '$NODES'"

require_docker

info "starting $NODES ${MPILAB_GPU:+GPU }node(s)"
compose up -d --scale "$SERVICE=$NODES" --remove-orphans

# The image declares HEALTHCHECK nc -z 127.0.0.1 22, so "healthy" means sshd
# has bound the port -- which is precisely the precondition mpirun needs.
info "waiting for sshd on every node"
deadline=$(( $(date +%s) + 120 ))
while :; do
    ids=$(node_ids)
    ready=0
    total=0
    for id in $ids; do
        total=$((total + 1))
        [ "$(node_health "$id")" = "healthy" ] && ready=$((ready + 1))
    done
    if [ "$total" -eq "$NODES" ] && [ "$ready" -eq "$NODES" ]; then
        break
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
        printf '\n' >&2
        die "timed out with $ready/$NODES node(s) healthy. Try: docker compose logs $SERVICE"
    fi
    printf '.' >&2
    sleep 1
done
printf '\n' >&2

"$REPO_ROOT/scripts/status.sh"

cat >&2 <<EOF

Ready. Try:
  ./scripts/run-mpi.sh -n $NODES examples/hello-mpi
EOF
