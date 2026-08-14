#!/bin/sh
# Turn a base image into a simulated MPI node. Run once, at image build time,
# from both Dockerfile and Dockerfile.cuda.
#
# This lives in one file rather than being written twice because the two images
# must agree exactly: they are the CPU and GPU flavours of the same node, and a
# difference in the ssh setup between them would show up as a cluster that
# works until you switch to the GPU overlay. The PAM subtlety below in
# particular is not something to rediscover twice.
#
# Expects the packages (openssh-server, and an MPI) to be installed already,
# and docker/ssh_config to have been copied to /etc/mpi-lab-ssh_config.

set -eu

MPI_USER=mpi
MPI_UID=1001

useradd --create-home --shell /bin/bash --uid "$MPI_UID" "$MPI_USER"

# ---------------------------------------------------------------------------
# SSH, and why a private key is baked into the image
# ---------------------------------------------------------------------------
# mpirun starts remote processes over ssh, so every node must be able to log
# into every other node without a prompt. The usual way to arrange that is for
# the user to generate a key and distribute it -- exactly the setup step this
# project exists to remove.
#
# Because every node runs the *same image*, one key pair generated here is
# already present and already authorised everywhere, and the cluster works on
# first boot with nothing for the user to do.
#
# The trade is real and bounded: anyone holding this image holds the key. That
# is acceptable only because of how these containers are used -- an internal
# bridge network, no published ports, nothing in them but example source.
#
#   *** Do not push an image built from the dev stage to a registry, and do
#   *** not reuse this key pair for anything that is not a throwaway container.
#
# docs/ARCHITECTURE.md covers the alternatives that were considered.
ssh-keygen -A
mkdir -p "/home/$MPI_USER/.ssh"
ssh-keygen -t ed25519 -N "" -C "mpi-local-lab dev key -- not for real hosts" \
           -f "/home/$MPI_USER/.ssh/id_ed25519"
cp "/home/$MPI_USER/.ssh/id_ed25519.pub" "/home/$MPI_USER/.ssh/authorized_keys"
cp /etc/mpi-lab-ssh_config "/home/$MPI_USER/.ssh/config"
chown -R "$MPI_USER:$MPI_USER" "/home/$MPI_USER/.ssh"
chmod 700 "/home/$MPI_USER/.ssh"
chmod 600 "/home/$MPI_USER/.ssh/id_ed25519" \
          "/home/$MPI_USER/.ssh/authorized_keys" \
          "/home/$MPI_USER/.ssh/config"

# Key-only, non-root. No account here has a password, so password auth could
# not succeed anyway -- turning it off makes that explicit rather than lucky.
#
# UsePAM is left at the distro default of yes, and that is load-bearing rather
# than lazy. useradd leaves the account's password field locked ("!"), and with
# UsePAM no, sshd runs its own locked-account check and refuses the login
# before it ever looks at the key: "User mpi not allowed because account is
# locked", which reaches the user as an unexplained "Permission denied
# (publickey)" and an mpirun that cannot reach a single node. With PAM in the
# loop that check belongs to pam_unix, which correctly allows key-based login
# to a password-locked account -- the arrangement every cloud image ships.
mkdir -p /etc/ssh/sshd_config.d
printf '%s\n' \
    'PermitRootLogin no' \
    'PasswordAuthentication no' \
    'KbdInteractiveAuthentication no' \
    'PermitEmptyPasswords no' \
    'X11Forwarding no' \
    "AllowUsers $MPI_USER" \
    > /etc/ssh/sshd_config.d/mpi-lab.conf

# Ubuntu 24.04's sshd_config reads the drop-in directory; older bases do not.
if ! grep -q '^[[:space:]]*Include[[:space:]]\+/etc/ssh/sshd_config.d/' /etc/ssh/sshd_config; then
    sed -i '1i Include /etc/ssh/sshd_config.d/*.conf' /etc/ssh/sshd_config
fi

# Both are mounts at run time; created here so a container started without
# compose still has somewhere to put things.
mkdir -p /workspace /build
chown "$MPI_USER:$MPI_USER" /workspace /build

rm -f /etc/mpi-lab-ssh_config
