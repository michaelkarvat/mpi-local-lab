# Shared helpers for the cluster scripts. Sourced, not executed.

set -euo pipefail

# ---------------------------------------------------------------------------
# Git Bash / MSYS argument mangling
# ---------------------------------------------------------------------------
# On Windows, MSYS rewrites any argument that looks like a POSIX path into a
# Windows one before the program sees it. `docker exec node ls /build` arrives
# as `ls C:/Program Files/Git/build`, and the error it produces points at the
# container rather than at the shell that caused it. These two variables turn
# the rewriting off for everything these scripts run.
export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL='*'

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SERVICE=node
BUILD_DIR=/build

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
info() { printf '[mpi-lab] %s\n' "$*" >&2; }

# Compose, always from the repository root so the project name and the relative
# bind mount in compose.yaml resolve the same way from any working directory.
#
# MPILAB_GPU=1 layers compose.gpu.yaml on top, which swaps the node image for
# the CUDA one. The compose project name is the same either way, so `ps` and
# `exec` work against a GPU cluster whether or not the variable is set -- it
# only has to be right for `up`.
compose() {
    local files=(-f compose.yaml)
    if [ "${MPILAB_GPU:-0}" = "1" ]; then
        files+=(-f compose.gpu.yaml)
    fi
    ( cd "$REPO_ROOT" && docker compose "${files[@]}" "$@" )
}

require_docker() {
    command -v docker >/dev/null 2>&1 || die "docker not found on PATH"
    docker info >/dev/null 2>&1 || die "cannot talk to the Docker daemon -- is Docker Desktop running?"
    docker compose version >/dev/null 2>&1 || die "this needs Docker Compose v2 (\`docker compose\`, not \`docker-compose\`)"
}

# Container ids of the running nodes, one per line, in creation order.
node_ids() {
    compose ps -q "$SERVICE" 2>/dev/null || true
}

node_count() {
    node_ids | grep -c . || true
}

require_cluster() {
    local n
    n=$(node_count)
    [ "$n" -gt 0 ] || die "no nodes are running. Start them with: ./scripts/start-cluster.sh"
}

# Run a command on node 1, as the mpi user. Node 1 is the head node by
# convention only -- every node is the same image and any of them would do.
on_head() {
    compose exec -T --index=1 -u mpi "$SERVICE" "$@"
}

# One field from `docker inspect`, for a container id.
inspect() {
    docker inspect -f "$2" "$1" 2>/dev/null || echo "?"
}

# A compact container-name -> hostname map, on one line.
#
# Worth printing before every run because a replica's hostname is its container
# id, and that id is what MPI_Get_processor_name reports. Without this line the
# output of hello-mpi proves the ranks landed on four different nodes but says
# nothing about *which*; with it, the run is readable on its own.
node_legend() {
    local out="" id
    for id in $(node_ids); do
        out="$out $(node_name "$id" | sed 's/^mpi-lab-//')=$(node_hostname "$id")"
    done
    printf '%s\n' "${out# }"
}

node_ip()       { inspect "$1" '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}'; }
node_hostname() { inspect "$1" '{{.Config.Hostname}}'; }
node_name()     { inspect "$1" '{{.Name}}' | sed 's#^/##'; }
node_state()    { inspect "$1" '{{.State.Status}}'; }
node_health()   { inspect "$1" '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}'; }
