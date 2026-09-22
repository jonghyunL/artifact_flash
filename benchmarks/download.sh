#!/usr/bin/env bash
#
# download.sh — fetch the ShareGPT dataset and the three models used in the
# evaluation (Llama-3.2-1B, Qwen-2.5-7B, Llama-2-13B).
#
# Usage:
#   ./download.sh                 # dataset + all three models
#   ./download.sh sharegpt        # dataset only
#   ./download.sh llama1b qwen7b  # selected models only
#
# The Llama models are gated on Hugging Face. Request access on each model page
# first, then log in once with `hf auth login` (or `huggingface-cli login`), or
# export HF_TOKEN=<token> before running this script.
#
# Environment overrides:
#   MODEL_DIR     where models go         (default ~/huggingface)
#   SHAREGPT_DIR  where the dataset goes  (default ~, the location
#                 bench_inference_vllm_async.py searches)

set -euo pipefail

MODEL_DIR="${MODEL_DIR:-$HOME/huggingface}"
SHAREGPT_DIR="${SHAREGPT_DIR:-$HOME}"

SHAREGPT_FILE="ShareGPT_V3_unfiltered_cleaned_split.json"
SHAREGPT_REPO="anon8231489123/ShareGPT_Vicuna_unfiltered"

declare -A MODELS=(
    [llama1b]="meta-llama/Llama-3.2-1B-Instruct"
    [qwen7b]="Qwen/Qwen2.5-7B-Instruct"
    [llama13b]="meta-llama/Llama-2-13b-chat-hf"
)
ORDER=(llama1b qwen7b llama13b)

if command -v hf >/dev/null 2>&1; then
    HF=(hf download)
elif command -v huggingface-cli >/dev/null 2>&1; then
    HF=(huggingface-cli download)
else
    echo "installing huggingface_hub CLI"
    pip install -U "huggingface_hub[cli]"
    HF=(huggingface-cli download)
fi

get_sharegpt() {
    local dst="$SHAREGPT_DIR/$SHAREGPT_FILE"
    if [[ -s "$dst" ]]; then
        echo "[sharegpt] already present: $dst"
        return
    fi
    echo "[sharegpt] -> $dst"
    mkdir -p "$SHAREGPT_DIR"
    "${HF[@]}" "$SHAREGPT_REPO" "$SHAREGPT_FILE" \
        --repo-type dataset --local-dir "$SHAREGPT_DIR"
}

get_model() {
    local repo="${MODELS[$1]}"
    local dst="$MODEL_DIR/${repo##*/}"
    echo "[$1] $repo -> $dst"
    mkdir -p "$dst"
    # Safetensors only; skip the duplicate PyTorch .bin / original/ checkpoints.
    "${HF[@]}" "$repo" --local-dir "$dst" \
        --exclude "*.bin" "*.pth" "original/*"
}

targets=("$@")
[[ ${#targets[@]} -eq 0 ]] && targets=(sharegpt "${ORDER[@]}")

for t in "${targets[@]}"; do
    case "$t" in
        sharegpt) get_sharegpt ;;
        llama1b|qwen7b|llama13b) get_model "$t" ;;
        *) echo "unknown target '$t' (sharegpt, llama1b, qwen7b, llama13b)" >&2; exit 1 ;;
    esac
done

echo
echo "done. models are in $MODEL_DIR, ShareGPT is in $SHAREGPT_DIR"
