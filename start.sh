#!/bin/bash
set -euo pipefail

# Do NOT import torch here in a throwaway process — that doubles cold-start
# time. rp_handler.py imports torch once in the long-lived worker.

PY="/runpod-volume/uv-python/cpython-3.12.14-linux-x86_64-gnu/bin/python3.12"
SITE="/runpod-volume/venvs/moncusoai/lib/python3.12/site-packages"

SRC_MODEL="${MODEL_PATH:-/runpod-volume/myapp/models/moncusoai/wvl81}"
LOCAL_MODEL="${LOCAL_MODEL_PATH:-/local/models/wvl81}"
STAGE_MODEL="${STAGE_MODEL:-1}"

if [[ ! -x "$PY" ]]; then
  echo "FATAL: volume Python not found at $PY" >&2
  ls -la /runpod-volume/uv-python 2>&1 || true
  ls -la /runpod-volume/venvs 2>&1 || true
  exit 127
fi

if [[ ! -d "$SITE" ]]; then
  echo "FATAL: site-packages not found at $SITE" >&2
  exit 127
fi

export VIRTUAL_ENV="/runpod-volume/venvs/moncusoai"
export PYTHONPATH="$SITE${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export PRELOAD_MODEL="${PRELOAD_MODEL:-1}"
export STAGE_MODEL
export MODEL_PATH="$SRC_MODEL"
export LOCAL_MODEL_PATH="$LOCAL_MODEL"

stage_model_to_local() {
  if [[ ! -d "$SRC_MODEL" ]]; then
    echo "FATAL: source model missing: $SRC_MODEL" >&2
    exit 1
  fi

  mkdir -p "$LOCAL_MODEL"
  local marker="$LOCAL_MODEL/.stage_complete"
  local idx="$SRC_MODEL/model.safetensors.index.json"

  # Skip if prior stage in this container looks complete and index matches.
  if [[ -f "$marker" && -f "$LOCAL_MODEL/model.safetensors.index.json" ]]; then
    if cmp -s "$idx" "$LOCAL_MODEL/model.safetensors.index.json" 2>/dev/null; then
      local missing=0
      for f in "$SRC_MODEL"/model-*.safetensors; do
        [[ -e "$LOCAL_MODEL/$(basename "$f")" ]] || missing=1
      done
      if [[ "$missing" -eq 0 ]]; then
        echo "Local model already staged at $LOCAL_MODEL"
        return 0
      fi
    fi
  fi

  echo "Staging inference weights: $SRC_MODEL -> $LOCAL_MODEL"
  df -h / "$LOCAL_MODEL" 2>/dev/null || df -h / || true
  local t0=$SECONDS

  # Small config / tokenizer files (skip distillation + training junk)
  local small=(
    config.json
    generation_config.json
    model.safetensors.index.json
    preprocessor_config.json
    video_preprocessor_config.json
    tokenizer.json
    tokenizer_config.json
    special_tokens_map.json
    added_tokens.json
    vocab.json
    merges.txt
    chat_template.jinja
  )
  for f in "${small[@]}"; do
    if [[ -e "$SRC_MODEL/$f" ]]; then
      cp -f "$SRC_MODEL/$f" "$LOCAL_MODEL/$f"
    fi
  done

  # Parallel shard copy — sequential NFS reads beat random weight-load I/O
  local pids=()
  for f in "$SRC_MODEL"/model-*.safetensors; do
    [[ -e "$f" ]] || continue
    local base
    base="$(basename "$f")"
    echo "  copying $base ..."
    cp -f "$f" "$LOCAL_MODEL/$base" &
    pids+=($!)
  done
  local fail=0
  for pid in "${pids[@]:-}"; do
    wait "$pid" || fail=1
  done
  if [[ "$fail" -ne 0 ]]; then
    echo "FATAL: shard copy failed (disk full? need ~20GB free)" >&2
    df -h / || true
    rm -f "$marker"
    exit 1
  fi

  # Sanity: all shards present and non-empty
  for f in "$SRC_MODEL"/model-*.safetensors; do
    local base size
    base="$(basename "$f")"
    size="$(stat -c%s "$LOCAL_MODEL/$base" 2>/dev/null || echo 0)"
    if [[ "$size" -lt 1000000 ]]; then
      echo "FATAL: staged shard missing/short: $base ($size bytes)" >&2
      exit 1
    fi
  done

  touch "$marker"
  echo "Staged in $((SECONDS - t0))s -> $LOCAL_MODEL ($(du -sh "$LOCAL_MODEL" | awk '{print $1}'))"
}

if [[ "$STAGE_MODEL" == "1" || "$STAGE_MODEL" == "true" ]]; then
  stage_model_to_local
  # Handler loads from local path when present
  export MODEL_LOAD_PATH="$LOCAL_MODEL"
else
  export MODEL_LOAD_PATH="$SRC_MODEL"
fi

echo "Starting worker: $PY  PRELOAD_MODEL=$PRELOAD_MODEL  MODEL_LOAD_PATH=$MODEL_LOAD_PATH"
exec "$PY" -u /rp_handler.py
