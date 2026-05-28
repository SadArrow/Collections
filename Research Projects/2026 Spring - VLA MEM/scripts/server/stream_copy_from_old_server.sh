#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 2 ]]; then
  echo "usage: $0 <src_path_on_old_server> <dest_parent_on_new_server>" >&2
  exit 2
fi

SRC_PATH="$1"
DEST_PARENT="$2"
OLD_HOST="${OLD_HOST:-27.190.15.150}"
OLD_USER="${OLD_USER:-ubuntu}"
ASKPASS_PATH="${ASKPASS_PATH:-/root/workspace/qianyupeng/tmp/oldserver_askpass.sh}"

if [[ ! -x "${ASKPASS_PATH}" ]]; then
  echo "missing executable askpass helper: ${ASKPASS_PATH}" >&2
  exit 2
fi

mkdir -p "${DEST_PARENT}"

SRC_PARENT="$(dirname "${SRC_PATH}")"
SRC_BASENAME="$(basename "${SRC_PATH}")"

export DISPLAY=:0
export SSH_ASKPASS="${ASKPASS_PATH}"

setsid -w ssh \
  -o StrictHostKeyChecking=no \
  -o PreferredAuthentications=password \
  -o PubkeyAuthentication=no \
  "${OLD_USER}@${OLD_HOST}" \
  "cd '${SRC_PARENT}' && tar -cf - '${SRC_BASENAME}'" \
  </dev/null \
  | tar -xpf - -C "${DEST_PARENT}"
