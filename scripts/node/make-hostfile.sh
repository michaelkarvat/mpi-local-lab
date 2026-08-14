#!/bin/sh
# Runs INSIDE a node. Writes an MPI hostfile describing the live cluster.
#
#   sh /workspace/scripts/node/make-hostfile.sh [outfile]
#
# The node list comes from Docker's embedded DNS: the compose service name
# `node` resolves to every replica's address, so the cluster describes itself
# and nothing on the host has to inspect containers and pass a list in. That is
# also why scaling the service up or down needs no change here -- the next run
# simply resolves a different number of addresses.
#
# Addresses rather than names because a replica's hostname is its container id,
# which resolves from the host but not necessarily from a peer.
#
# Prints a one-line summary on stdout for the caller to parse.

set -eu

OUT=${1:-/build/hostfile}
SLOTS=$(nproc)

# Numeric sort on each octet, so the ordering is stable between runs and
# .10 does not sort before .2. Rank placement follows hostfile order, so a
# stable list means rank 0 lands on the same node every time.
ips=$(getent ahostsv4 node | awk '{print $1}' | sort -u \
        | sort -t. -k1,1n -k2,2n -k3,3n -k4,4n)

[ -n "$ips" ] || { echo "make-hostfile: service 'node' resolved to nothing" >&2; exit 1; }

: > "$OUT"
count=0
for ip in $ips; do
    echo "$ip slots=$SLOTS" >> "$OUT"
    count=$((count + 1))
done

echo "nodes=$count slots_per_node=$SLOTS total_slots=$((count * SLOTS)) hostfile=$OUT"
