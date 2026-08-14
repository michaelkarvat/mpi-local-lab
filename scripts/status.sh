#!/usr/bin/env bash
# What the cluster currently looks like.
#
#   ./scripts/status.sh
#
# The HOSTNAME column is the useful one. Because the nodes are replicas of one
# compose service, each container's hostname is its container id, and that id
# is what MPI_Get_processor_name reports -- so this table is how you read the
# output of examples/hello-mpi and see which node each rank landed on.

set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

require_docker

ids=$(node_ids)
if [ -z "$ids" ]; then
    echo "no nodes running. Start them with: ./scripts/start-cluster.sh"
    exit 0
fi

printf '%-18s  %-14s  %-13s  %-9s  %s\n' NODE HOSTNAME IP STATE HEALTH
for id in $ids; do
    printf '%-18s  %-14s  %-13s  %-9s  %s\n' \
        "$(node_name "$id")" \
        "$(node_hostname "$id")" \
        "$(node_ip "$id")" \
        "$(node_state "$id")" \
        "$(node_health "$id")"
done

echo
echo "nodes: $(node_count)   network: mpi-lab-net   build volume: mpi-lab-build"
