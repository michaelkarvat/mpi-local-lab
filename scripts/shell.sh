#!/usr/bin/env bash
# Open an interactive shell on one of the nodes.
#
#   ./scripts/shell.sh        # node 1
#   ./scripts/shell.sh 3      # node 3
#
# You land in /workspace as the `mpi` user, with the toolchain, mpirun, gdb and
# the source tree all present. This is the place to run one-off commands, debug
# a rank under gdb, or ssh to a peer by hand to see the launcher's view of the
# network.

set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

INDEX=${1:-1}
[[ "$INDEX" =~ ^[0-9]+$ ]] && [ "$INDEX" -ge 1 ] || die "node index must be a positive integer, got '$INDEX'"

require_docker
require_cluster

n=$(node_count)
[ "$INDEX" -le "$n" ] || die "node $INDEX does not exist; the cluster has $n node(s)"

# No -T here: this one wants the TTY.
( cd "$REPO_ROOT" && docker compose exec --index="$INDEX" -u mpi "$SERVICE" bash )
