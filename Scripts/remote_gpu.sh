#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  BRUSHWRIGHT_REMOTE=root@HOST [BRUSHWRIGHT_REMOTE_DIR=/workspace/BrushWright] Scripts/remote_gpu.sh COMMAND [ARGS...]

Commands:
  check           Print remote cwd, Python version, and GPU summary.
  push-code        Sync source, docs, configs, tests, scripts, fixtures, and README files.
  push-data        Sync generated/local training inputs: Data, Assets/ImageCorpus, and ThirdParty/PaintTransformer.
  push-all         Run push-code and push-data.
  train [ARGS...]  Sync code, ensure a remote venv, then run the V8 large visual-delta trainer.
  pull-artifacts   Pull Models/Checkpoints and Outputs back from the remote.
  shell            Open an SSH shell in the remote BrushWright directory.

Examples:
  BRUSHWRIGHT_REMOTE=root@203.0.113.10 Scripts/remote_gpu.sh push-all
  BRUSHWRIGHT_REMOTE=root@203.0.113.10 Scripts/remote_gpu.sh train --epochs 80
  BRUSHWRIGHT_REMOTE=root@203.0.113.10 Scripts/remote_gpu.sh pull-artifacts

Optional environment:
  BRUSHWRIGHT_REMOTE_DIR   Remote checkout path. Default: /workspace/BrushWright
  BRUSHWRIGHT_SSH_OPTS     Extra ssh options, for example: -i ~/.ssh/gpu_key -p 2222
  BRUSHWRIGHT_RSYNC_OPTS   Extra rsync options.
  BRUSHWRIGHT_SSH_FORCE_TTY Set to 1 for PTY-only SSH gateways.
  BRUSHWRIGHT_SYNC_DATA    Set to 1 for train to run push-data before training.
EOF
}

require_remote() {
  if [[ -z "${BRUSHWRIGHT_REMOTE:-}" ]]; then
    echo "BRUSHWRIGHT_REMOTE is required, for example root@203.0.113.10" >&2
    exit 2
  fi
}

repo_root() {
  cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd
}

remote_dir() {
  printf '%s\n' "${BRUSHWRIGHT_REMOTE_DIR:-/workspace/BrushWright}"
}

validate_remote_dir() {
  local dir
  dir="$(remote_dir)"
  case "$dir" in
    /|/root|/home|/workspace|"")
      echo "Refusing unsafe BRUSHWRIGHT_REMOTE_DIR: $dir" >&2
      exit 2
      ;;
  esac
}

ssh_remote() {
  # BRUSHWRIGHT_SSH_OPTS is intentionally split so users can pass normal ssh flags.
  # shellcheck disable=SC2086
  ssh ${BRUSHWRIGHT_SSH_OPTS:-} "$BRUSHWRIGHT_REMOTE" "$@"
}

ssh_remote_tty_stdin() {
  # BRUSHWRIGHT_SSH_OPTS is intentionally split so users can pass normal ssh flags.
  # shellcheck disable=SC2086
  ssh ${BRUSHWRIGHT_SSH_OPTS:-} -tt "$BRUSHWRIGHT_REMOTE"
}

remote_exec() {
  local command="$1"
  if [[ "${BRUSHWRIGHT_SSH_FORCE_TTY:-0}" == "1" ]]; then
    printf '%s\nexit\n' "$command" | ssh_remote_tty_stdin
  else
    ssh_remote "$command"
  fi
}

rsync_remote() {
  local source_path="$1"
  local target_path="$2"
  shift 2
  # BRUSHWRIGHT_RSYNC_OPTS and BRUSHWRIGHT_SSH_OPTS are intentionally split for CLI-style flags.
  # shellcheck disable=SC2086
  rsync -az --info=progress2 ${BRUSHWRIGHT_RSYNC_OPTS:-} \
    -e "ssh ${BRUSHWRIGHT_SSH_OPTS:-}" \
    "$@" "$source_path" "$BRUSHWRIGHT_REMOTE:$target_path"
}

rsync_from_remote() {
  local source_path="$1"
  local target_path="$2"
  # BRUSHWRIGHT_RSYNC_OPTS and BRUSHWRIGHT_SSH_OPTS are intentionally split for CLI-style flags.
  # shellcheck disable=SC2086
  rsync -az --info=progress2 ${BRUSHWRIGHT_RSYNC_OPTS:-} \
    -e "ssh ${BRUSHWRIGHT_SSH_OPTS:-}" \
    "$BRUSHWRIGHT_REMOTE:$source_path" "$target_path"
}

ensure_remote_dir() {
  validate_remote_dir
  remote_exec "mkdir -p '$(remote_dir)'"
}

push_code() {
  require_remote
  ensure_remote_dir
  if [[ "${BRUSHWRIGHT_SSH_FORCE_TTY:-0}" == "1" ]]; then
    push_code_archive
    return
  fi
  local root
  root="$(repo_root)"
  rsync_remote "$root/" "$(remote_dir)/" \
    --exclude ".git/" \
    --exclude ".agents/" \
    --exclude ".codex/" \
    --exclude ".venv/" \
    --exclude ".vscode/" \
    --exclude "__pycache__/" \
    --exclude "*.pyc" \
    --exclude "Data/Train/" \
    --exclude "Data/Val/" \
    --exclude "Data/Test/" \
    --exclude "Data/dataset_manifest.json" \
    --include "Outputs/README.md" \
    --exclude "Outputs/*" \
    --exclude "Models/Checkpoints/*" \
    --include "Assets/ImageCorpus/README.md" \
    --exclude "Assets/ImageCorpus/*" \
    --exclude "ThirdParty/PaintTransformer/*.pth"
}

push_code_archive() {
  local root archive remote_archive
  root="$(repo_root)"
  archive="/tmp/brushwright_code_$$.tar.gz"
  remote_archive="/tmp/brushwright_code_$$.tar.gz"
  (
    cd "$root"
    tar -czf "$archive" \
      --exclude "__pycache__" \
      --exclude "*.pyc" \
      --exclude "Models/Checkpoints/*" \
      --exclude "ThirdParty/PaintTransformer/*.pth" \
      AGENTS.md \
      NOTICE.md \
      README.md \
      STYLE_GUIDE.md \
      run.py \
      .gitignore \
      Config \
      Docs \
      Fixtures \
      Scripts \
      Source \
      Tests \
      ThirdParty \
      Assets/README.md \
      Assets/ImageCorpus/README.md \
      Data/README.md \
      Outputs/README.md \
      Models
  )
  {
    printf "stty -echo 2>/dev/null || true\n"
    printf "mkdir -p '%s'\n" "$(remote_dir)"
    printf "base64 -d > '%s' <<'BRUSHWRIGHT_ARCHIVE'\n" "$remote_archive"
    base64 "$archive"
    printf "BRUSHWRIGHT_ARCHIVE\n"
    printf "tar --no-same-owner -xzf '%s' -C '%s'\n" "$remote_archive" "$(remote_dir)"
    printf "rm -f '%s'\n" "$remote_archive"
    printf "stty echo 2>/dev/null || true\n"
    printf "exit\n"
  } | ssh_remote_tty_stdin
  rm -f "$archive"
}

push_data() {
  require_remote
  ensure_remote_dir
  if [[ "${BRUSHWRIGHT_SSH_FORCE_TTY:-0}" == "1" ]]; then
    push_data_archive
    return
  fi
  local root
  root="$(repo_root)"
  mkdir -p "$root/Data" "$root/Assets/ImageCorpus" "$root/ThirdParty/PaintTransformer"
  rsync_remote "$root/Data/" "$(remote_dir)/Data/"
  rsync_remote "$root/Assets/ImageCorpus/" "$(remote_dir)/Assets/ImageCorpus/"
  rsync_remote "$root/ThirdParty/PaintTransformer/" "$(remote_dir)/ThirdParty/PaintTransformer/"
}

push_data_archive() {
  local root archive remote_archive
  root="$(repo_root)"
  archive="/tmp/brushwright_data_$$.tar.gz"
  remote_archive="/tmp/brushwright_data_$$.tar.gz"
  (
    cd "$root"
    tar --ignore-failed-read -czf "$archive" \
      Data \
      Assets/ImageCorpus \
      ThirdParty/PaintTransformer
  )
  {
    printf "stty -echo 2>/dev/null || true\n"
    printf "mkdir -p '%s'\n" "$(remote_dir)"
    printf "base64 -d > '%s' <<'BRUSHWRIGHT_ARCHIVE'\n" "$remote_archive"
    base64 "$archive"
    printf "BRUSHWRIGHT_ARCHIVE\n"
    printf "tar --no-same-owner -xzf '%s' -C '%s'\n" "$remote_archive" "$(remote_dir)"
    printf "rm -f '%s'\n" "$remote_archive"
    printf "stty echo 2>/dev/null || true\n"
    printf "exit\n"
  } | ssh_remote_tty_stdin
  rm -f "$archive"
}

remote_bootstrap() {
  require_remote
  ensure_remote_dir
  remote_exec "cd '$(remote_dir)' && if [ ! -x .venv/bin/python ]; then python3 -m venv .venv; fi && (.venv/bin/python -c 'import torch, PIL, numpy') || (.venv/bin/python -m pip install --upgrade pip && .venv/bin/python -m pip install torch pillow numpy)"
}

train_remote() {
  require_remote
  push_code
  if [[ "${BRUSHWRIGHT_SYNC_DATA:-0}" == "1" ]]; then
    push_data
  fi
  remote_bootstrap
  local args=("$@")
  if [[ "${#args[@]}" -eq 0 ]]; then
    args=(--device cuda --visual-validation-device cuda)
  fi
  local remote_args
  printf -v remote_args '%q ' "${args[@]}"
  remote_exec "cd '$(remote_dir)' && .venv/bin/python -m Source.Model.train_visual_delta_strokes ${remote_args}"
}

pull_artifacts() {
  require_remote
  validate_remote_dir
  local root
  root="$(repo_root)"
  mkdir -p "$root/Models/Checkpoints" "$root/Outputs"
  rsync_from_remote "$(remote_dir)/Models/Checkpoints/" "$root/Models/Checkpoints/"
  rsync_from_remote "$(remote_dir)/Outputs/" "$root/Outputs/"
}

open_shell() {
  require_remote
  ensure_remote_dir
  # BRUSHWRIGHT_SSH_OPTS is intentionally split so users can pass normal ssh flags.
  # shellcheck disable=SC2086
  ssh ${BRUSHWRIGHT_SSH_OPTS:-} -tt "$BRUSHWRIGHT_REMOTE" "cd '$(remote_dir)' && exec bash"
}

check_remote() {
  require_remote
  remote_exec "cd '$(remote_dir)' 2>/dev/null || pwd; pwd; python3 --version; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"
}

main() {
  local command="${1:-}"
  if [[ -z "$command" || "$command" == "-h" || "$command" == "--help" ]]; then
    usage
    exit 0
  fi
  shift
  case "$command" in
    check) check_remote "$@" ;;
    push-code) push_code "$@" ;;
    push-data) push_data "$@" ;;
    push-all) push_code && push_data ;;
    train) train_remote "$@" ;;
    pull-artifacts) pull_artifacts "$@" ;;
    shell) open_shell "$@" ;;
    *)
      echo "Unknown command: $command" >&2
      usage >&2
      exit 2
      ;;
  esac
}

main "$@"
