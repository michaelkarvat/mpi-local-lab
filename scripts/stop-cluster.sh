#!/usr/bin/env bash
# Tear the cluster down.
#
#   ./scripts/stop-cluster.sh           # stop and remove the containers
#   ./scripts/stop-cluster.sh --purge   # also delete the shared build volume
#
# Without --purge the build volume survives, so the next start-cluster does an
# incremental build instead of compiling the tree again. Use --purge when you
# want a genuinely clean build, or when a configure has gone wrong and you want
# CMake's cache gone with it.

set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

PURGE=0
case "${1:-}" in
    --purge|-p) PURGE=1 ;;
    "") ;;
    *) die "unknown argument '$1' (expected --purge)" ;;
esac

require_docker

if [ "$PURGE" -eq 1 ]; then
    info "stopping the cluster and removing the build volume"
    compose down --remove-orphans --volumes
else
    info "stopping the cluster (build volume kept; --purge removes it)"
    compose down --remove-orphans
fi
