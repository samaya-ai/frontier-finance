#!/usr/bin/env bash
#
# Download the FrontierFinance rubrics from Hugging Face into the current
# directory.
#
# The dataset ships the rubrics only (frontier_finance_public.jsonl); you supply
# your own system responses to grade against them.
#
# Usage:
#   ./scripts/download_data.sh [DEST_DIR]
#
# DEST_DIR defaults to the current working directory.
#
set -euo pipefail

REPO_ID="samaya-ai/FrontierFinance"
REVISION="main"
FILE="frontier_finance_public.jsonl"

DEST_DIR="${1:-.}"

# Public dataset, so no token is needed; honor one if it happens to be set.
TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"

mkdir -p "${DEST_DIR}"

echo "Downloading ${REPO_ID}/${FILE} -> ${DEST_DIR}/"

# Prefer the official CLI when present: it handles resume and verification.
if command -v hf >/dev/null 2>&1; then
  echo "Using hf CLI"
  args=(download "${REPO_ID}" "${FILE}"
        --repo-type dataset --revision "${REVISION}"
        --local-dir "${DEST_DIR}")
  [ -n "${TOKEN}" ] && args+=(--token "${TOKEN}")
  hf "${args[@]}"
  echo "Done -> ${DEST_DIR}/${FILE}"
  exit 0
fi

# Fallback: curl the resolve endpoint directly (no Python deps required).
echo "hf CLI not found; falling back to curl"
auth=()
[ -n "${TOKEN}" ] && auth=(-H "Authorization: Bearer ${TOKEN}")

curl -fL "${auth[@]}" --progress-bar \
  -o "${DEST_DIR}/${FILE}" \
  "https://huggingface.co/datasets/${REPO_ID}/resolve/${REVISION}/${FILE}"

echo "Done -> ${DEST_DIR}/${FILE}"
