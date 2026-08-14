#!/usr/bin/env bash
# Compile an example and run it across the cluster.
#
#   ./scripts/run-mpi.sh -n 4 examples/hello-mpi
#   ./scripts/run-mpi.sh -n 8 distributed-sum 5000000
#   ./scripts/run-mpi.sh -n 4 --no-build matrix-multiply 512
#
# -n is the number of RANKS, not nodes. The node count is however many
# containers are running; change it with ./scripts/start-cluster.sh N.
#
# Ranks are placed round-robin over the nodes (--map-by node), so `-n 4` on a
# four-node cluster puts exactly one rank on each -- the arrangement the
# architecture diagram in the README shows -- while `-n 8` gives two per node.
# The alternative, filling each node's slots before moving to the next, would
# put all four ranks of a small run on node 1 and quietly stop exercising the
# network at all.

set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

RANKS=""
BUILD=1

usage() {
    sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

while [ $# -gt 0 ]; do
    case "$1" in
        -n|--ranks)  RANKS=${2:-}; shift 2 ;;
        -n*)         RANKS=${1#-n}; shift ;;
        --no-build|-B) BUILD=0; shift ;;
        -h|--help)   usage 0 ;;
        --)          shift; break ;;
        -*)          die "unknown option '$1' (try --help)" ;;
        *)           break ;;
    esac
done

[ $# -ge 1 ] || usage 1
require_docker
require_cluster

# Accept every spelling the docs and the PRD use: a path as it appears in the
# tree, with or without the examples/ prefix or a trailing slash, or the bare
# target name. They all denote the same CMake target.
TARGET=${1%/}
TARGET=${TARGET#./}
TARGET=${TARGET#examples/}
TARGET=${TARGET%%/*}
shift

NODES=$(node_count)
[ -n "$RANKS" ] || RANKS=$NODES
[[ "$RANKS" =~ ^[0-9]+$ ]] && [ "$RANKS" -ge 1 ] || die "rank count must be a positive integer, got '$RANKS'"

# Resolve the name the user typed to the target that actually gets built, and
# compile it. Both happen in one call because resolution needs the configure
# step to have run. --no-build still resolves: skipping the compiler is not the
# same as skipping the lookup.
build_env=()
[ "$BUILD" -eq 1 ] || build_env=(-e MPILAB_NO_COMPILE=1)

build_log=$(compose exec -T --index=1 -u mpi "${build_env[@]}" "$SERVICE" \
                sh /workspace/scripts/node/build-example.sh "$TARGET" 2>&1) \
    || { printf '%s\n' "$build_log" >&2; die "could not build '$TARGET'"; }

# Compiler output belongs on stderr so a piped stdout carries only the
# program's own output.
printf '%s\n' "$build_log" | grep -v '^mpilab-target=' >&2 || true
BINARY=$(printf '%s\n' "$build_log" | sed -n 's/^mpilab-target=//p' | tail -1)
[ -n "$BINARY" ] || BINARY=$TARGET

# Checked here rather than left to mpirun. Its failure for a missing executable
# is a twelve-line box that names the path and the node but not the reason, and
# with --no-build the reason is almost always that this target has simply never
# been compiled into the shared volume.
if ! on_head test -x "$BUILD_DIR/bin/$BINARY" 2>/dev/null; then
    if [ "$BUILD" -eq 0 ]; then
        die "$BUILD_DIR/bin/$BINARY does not exist, and --no-build was given. Drop --no-build to compile it."
    fi
    die "$BUILD_DIR/bin/$BINARY does not exist even after building '$TARGET'."
fi

# Regenerated every run: the cluster may have been scaled since the last one,
# and a stale hostfile naming a departed node hangs the launch.
summary=$(on_head sh /workspace/scripts/node/make-hostfile.sh "$BUILD_DIR/hostfile")
total_slots=$(printf '%s\n' "$summary" | sed -n 's/.*total_slots=\([0-9]*\).*/\1/p')

oversubscribe=()
if [ -n "$total_slots" ] && [ "$RANKS" -gt "$total_slots" ]; then
    info "$RANKS ranks over $total_slots slots -- oversubscribing (fine for correctness, meaningless for timing)"
    oversubscribe=(--oversubscribe)
fi

info "$NODES node(s), $RANKS rank(s) -> $BUILD_DIR/bin/$BINARY"
info "nodes: $(node_legend)"

on_head mpirun \
    -np "$RANKS" \
    --hostfile "$BUILD_DIR/hostfile" \
    --map-by node \
    "${oversubscribe[@]}" \
    "$BUILD_DIR/bin/$BINARY" "$@"
