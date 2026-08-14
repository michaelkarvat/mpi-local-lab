#!/usr/bin/env bash
# Environment tests: does the simulated cluster actually behave like a cluster?
#
#   ./scripts/start-cluster.sh 4
#   bash tests/cluster/run_all.sh
#
# These are deliberately not CTest tests. Everything under ctest builds and
# runs inside one container and would pass just as well on a laptop with no
# Docker at all; what is checked here is the part ctest cannot see -- that the
# nodes are separate machines as far as MPI is concerned, that they can find
# and authenticate to each other, and that a program launched on one of them
# really executes on all of them.
#
# The suite needs a running cluster and leaves it running.

set -uo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../scripts" && pwd)/lib.sh"

PASS=0
FAIL=0

ok()   { printf '  ok   %s\n' "$*"; PASS=$((PASS + 1)); }
bad()  { printf '  FAIL %s\n' "$*"; FAIL=$((FAIL + 1)); }
group(){ printf '\n%s\n' "$*"; }

# `set -e` is off on purpose: a failing check must record itself and let the
# rest run, so one broken thing does not hide five working ones.

require_docker
require_cluster

NODES=$(node_count)
IDS=$(node_ids)
printf 'cluster: %s node(s)\n' "$NODES"

# ---------------------------------------------------------------------------
group "0. build -- compile every example once, on node 1"
# ---------------------------------------------------------------------------
# One build up front so the launch checks below can use --no-build and measure
# the launch rather than the compiler. It also means a compile error is
# reported as a compile error instead of as five mysterious launch failures.
if build_out=$(compose exec -T --index=1 -u mpi "$SERVICE" \
                   sh /workspace/scripts/node/build-example.sh all 2>&1); then
    ok "every example built into the shared volume"
else
    bad "build failed"
    printf '%s\n' "$build_out" | tail -20 | sed 's/^/       /'
    printf '\npassed %d, failed %d\n' "$PASS" "$FAIL"
    exit 1
fi

# ---------------------------------------------------------------------------
group "1. name resolution -- every node can find every other node"
# ---------------------------------------------------------------------------
# The compose service name resolves to all replicas through Docker's embedded
# DNS. This is what scripts/node/make-hostfile.sh relies on, and if it returns
# fewer addresses than there are nodes, every later launch silently uses a
# short cluster.
i=0
for id in $IDS; do
    i=$((i + 1))
    seen=$(docker exec "$id" getent ahostsv4 node 2>/dev/null | awk '{print $1}' | sort -u | grep -c .)
    if [ "$seen" -eq "$NODES" ]; then
        ok "node-$i resolves 'node' to $seen address(es)"
    else
        bad "node-$i resolves 'node' to $seen address(es), expected $NODES"
    fi
done

# ---------------------------------------------------------------------------
group "2. ssh -- mpirun's launcher can reach every node without a prompt"
# ---------------------------------------------------------------------------
# BatchMode=yes turns any request for input into an immediate failure, so a
# missing key or an unknown host key fails here rather than hanging a launch.
peers=""
for id in $IDS; do
    peers="$peers $(node_ip "$id")"
done
for peer in $peers; do
    if out=$(compose exec -T --index=1 -u mpi "$SERVICE" \
                ssh -o BatchMode=yes -o ConnectTimeout=10 "$peer" hostname 2>&1); then
        ok "node-1 -> $peer ($(printf '%s' "$out" | tr -d '\r'))"
    else
        bad "node-1 -> $peer: $(printf '%s' "$out" | head -1)"
    fi
done

# ---------------------------------------------------------------------------
group "3. cross-container MPI -- ranks land on distinct nodes"
# ---------------------------------------------------------------------------
# The headline claim of the whole project. One rank per node, and every rank
# must report a different hostname; if the launcher quietly put them all on the
# head node this is the check that notices.
out=$("$REPO_ROOT/scripts/run-mpi.sh" -n "$NODES" --no-build hello-mpi 2>/dev/null)
lines=$(printf '%s\n' "$out" | grep -c '^Hello from rank')
hosts=$(printf '%s\n' "$out" | sed -n 's/.* on \(.*\)$/\1/p' | sort -u | grep -c .)

if [ "$lines" -eq "$NODES" ]; then
    ok "$NODES rank(s) reported"
else
    bad "expected $NODES lines from hello-mpi, got $lines"
    printf '%s\n' "$out" | sed 's/^/       /'
fi
if [ "$hosts" -eq "$NODES" ]; then
    ok "$hosts distinct hostname(s) -- the ranks really are on separate nodes"
else
    bad "expected $NODES distinct hostnames, got $hosts (ranks shared a node)"
    printf '%s\n' "$out" | sed 's/^/       /'
fi

# ---------------------------------------------------------------------------
group "4. rank counts -- the answer does not depend on how many ranks ran"
# ---------------------------------------------------------------------------
# The in-container version of this is a CTest test. Repeating it across real
# containers is what proves the collectives work over the bridge network and
# not just over shared memory.
baseline=""
for ranks in 1 2 "$NODES"; do
    got=$("$REPO_ROOT/scripts/run-mpi.sh" -n "$ranks" --no-build distributed-sum 100000 2>/dev/null)
    if [ -z "$got" ]; then
        bad "distributed-sum at $ranks rank(s) produced nothing"
        continue
    fi
    if [ -z "$baseline" ]; then
        baseline=$got
        ok "distributed-sum at $ranks rank(s) -> baseline"
    elif [ "$got" = "$baseline" ]; then
        ok "distributed-sum at $ranks rank(s) -> identical to baseline"
    else
        bad "distributed-sum at $ranks rank(s) differs from the 1-rank baseline"
    fi
done

# ---------------------------------------------------------------------------
group "5. the advanced example -- golden output, distributed over containers"
# ---------------------------------------------------------------------------
# examples/hybrid-object-matching guarantees byte-identical results for any
# rank count. Here that guarantee is checked against the committed golden file
# with the ranks spread over real containers.
expected="$REPO_ROOT/examples/hybrid-object-matching/tests/data/reference.expected"
input=examples/hybrid-object-matching/tests/data/reference.txt
actual=$(mktemp)
"$REPO_ROOT/scripts/run-mpi.sh" -n "$NODES" --no-build msearch \
    --quiet --backend serial -i "$input" -o - 2>/dev/null > "$actual"
if diff -q "$expected" "$actual" >/dev/null 2>&1; then
    ok "msearch over $NODES container(s) matches the golden output"
else
    bad "msearch over $NODES container(s) differs from the golden output"
    diff "$expected" "$actual" | head -10 | sed 's/^/       /'
fi
rm -f "$actual"

# ---------------------------------------------------------------------------
printf '\n%s\n' "-----------------------------------------"
printf 'passed %d, failed %d\n' "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ] || exit 1
