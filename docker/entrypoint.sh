#!/bin/sh
# Entrypoint for a simulated MPI node.
#
# The container's job is to sit there and accept ssh logins, because that is
# how mpirun starts processes on a remote node. Everything else -- compiling,
# running -- arrives later as `docker compose exec`.
#
# sshd is the foreground process rather than a background one under a sleep
# loop, so the container's lifetime is exactly sshd's lifetime: if it dies the
# container dies, and `docker compose ps` tells the truth.

set -e

# Required by sshd for privilege separation; not created by the package.
mkdir -p /run/sshd

# /build is a named volume shared by every node. Docker creates it root-owned
# on first use, and the compiler runs as `mpi`, so hand it over. Every node
# runs this and only the first one has any work to do -- chown on an
# already-correct directory is a no-op, so the race is harmless.
if [ -d /build ]; then
    chown mpi:mpi /build 2>/dev/null || true
fi

exec /usr/sbin/sshd -D -e "$@"
