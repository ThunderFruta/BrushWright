#!/usr/bin/env bash
set -euo pipefail

export BRUSHWRIGHT_REMOTE="${BRUSHWRIGHT_REMOTE:-eq39y5ydoccbct-64411cd6@ssh.runpod.io}"
export BRUSHWRIGHT_SSH_OPTS="${BRUSHWRIGHT_SSH_OPTS:--i ~/.ssh/id_ed25519}"
export BRUSHWRIGHT_REMOTE_DIR="${BRUSHWRIGHT_REMOTE_DIR:-/workspace/BrushWright}"
export BRUSHWRIGHT_SSH_FORCE_TTY="${BRUSHWRIGHT_SSH_FORCE_TTY:-1}"

exec "$(dirname "${BASH_SOURCE[0]}")/remote_gpu.sh" "$@"
