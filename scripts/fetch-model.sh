#!/usr/bin/env bash
# Download a public Hugging Face model repo with resumable transfers and LFS checksums.
set -euo pipefail

usage() {
  printf 'Usage: %s REPO DIR [--dry-run]\n' "$(basename "$0")" >&2
}

if [[ $# -lt 2 || $# -gt 3 ]]; then
  usage
  exit 2
fi

REPO="$1"
DIR="$2"
DRY_RUN=false
if [[ $# -eq 3 ]]; then
  if [[ "$3" != "--dry-run" ]]; then
    usage
    exit 2
  fi
  DRY_RUN=true
fi

if [[ -z "$REPO" || -z "$DIR" ]]; then
  usage
  exit 2
fi

if ! command -v python3 >/dev/null 2>&1; then
  printf 'python3 is required to parse the Hugging Face model API response.\n' >&2
  exit 1
fi
TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/fetch-model.XXXXXX")"
cleanup() {
  rm -rf -- "$TMP_DIR"
}
trap cleanup EXIT

RESPONSE="$TMP_DIR/model.json"
MANIFEST="$TMP_DIR/manifest.tsv"
ARIA2_LIST="$TMP_DIR/aria2.txt"
CHECKSUMS="$TMP_DIR/checksums.txt"

API_BASE="${HF_API_BASE:-https://huggingface.co}"
if [[ -n "${HF_API_FILE:-}" ]]; then
  if [[ ! -f "$HF_API_FILE" ]]; then
    printf 'HF_API_FILE does not point to a readable model metadata file: %s\n' "$HF_API_FILE" >&2
    exit 1
  fi
  if ! cat "$HF_API_FILE" > "$RESPONSE"; then
    printf 'Could not read model metadata from HF_API_FILE: %s\n' "$HF_API_FILE" >&2
    exit 1
  fi
  HTTP_STATUS=200
else
  if ! command -v curl >/dev/null 2>&1; then
    printf 'curl is required to query the Hugging Face model API.\n' >&2
    exit 1
  fi
  ENCODED_REPO="$(python3 -c 'from urllib.parse import quote; import sys; print(quote(sys.argv[1], safe="/"))' "$REPO")"
  API_URL="${API_BASE%/}/api/models/${ENCODED_REPO}?blobs=true"
  if ! HTTP_STATUS="$(curl --globoff --silent --show-error --location \
    --output "$RESPONSE" --write-out '%{http_code}' "$API_URL")"; then
    printf 'Could not retrieve model metadata for %s from Hugging Face.\n' "$REPO" >&2
    exit 1
  fi
fi

case "$HTTP_STATUS" in
  2??) ;;
  404)
    printf 'Model repo %s was not found or is gated/private (HTTP %s); only public ungated repos can be downloaded.\n' \
      "$REPO" "$HTTP_STATUS" >&2
    exit 1
    ;;
  401|403)
    printf 'Access to model repo %s is denied or gated (HTTP %s); only public ungated repos can be downloaded.\n' \
      "$REPO" "$HTTP_STATUS" >&2
    exit 1
    ;;
  *)
    printf 'Hugging Face returned HTTP %s while looking up model repo %s.\n' "$HTTP_STATUS" "$REPO" >&2
    exit 1
    ;;
esac

if ! python3 -c '
import json
import os
import sys

repo, response_path, manifest_path = sys.argv[1:]
try:
    with open(response_path, "r", encoding="utf-8") as response_file:
        payload = json.loads(response_file.read(), strict=False)
except (OSError, UnicodeError, ValueError) as exc:
    print("Could not parse Hugging Face model metadata for %s: %s" % (repo, exc), file=sys.stderr)
    sys.exit(1)

if isinstance(payload, dict) and payload.get("error"):
    print("Model repo %s was not found or is inaccessible/gated: %s" % (repo, payload["error"]), file=sys.stderr)
    sys.exit(1)
if not isinstance(payload, dict) or not isinstance(payload.get("siblings"), list):
    print("Hugging Face returned no file list for model repo %s." % repo, file=sys.stderr)
    sys.exit(1)
if not payload["siblings"]:
    print("Model repo %s contains no downloadable files." % repo, file=sys.stderr)
    sys.exit(1)

try:
    with open(manifest_path, "w", encoding="utf-8", newline="") as manifest:
        for entry in payload["siblings"]:
            if not isinstance(entry, dict):
                raise ValueError("invalid file entry")
            name = entry.get("rfilename")
            if not isinstance(name, str) or not name or "\t" in name or "\n" in name or "\r" in name:
                raise ValueError("file names must be nonempty and cannot contain tabs or newlines")
            if name.startswith("/") or any(part == ".." for part in name.split("/")):
                raise ValueError("unsafe repository file path: %r" % name)
            lfs = entry.get("lfs") or {}
            size = entry.get("size")
            if size is None and isinstance(lfs, dict):
                size = lfs.get("size")
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise ValueError("missing or invalid size for %r" % name)
            digest = ""
            if isinstance(lfs, dict) and lfs.get("sha256") is not None:
                digest = lfs["sha256"]
                if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdefABCDEF" for c in digest):
                    raise ValueError("invalid LFS sha256 for %r" % name)
            manifest.write("%s\t%d\t%s\n" % (name, size, digest))
except (OSError, ValueError) as exc:
    print("Invalid file metadata for model repo %s: %s" % (repo, exc), file=sys.stderr)
    sys.exit(1)
' "$REPO" "$RESPONSE" "$MANIFEST"; then
  exit 1
fi

: > "$ARIA2_LIST"
: > "$CHECKSUMS"
TOTAL_BYTES=0
FILE_COUNT=0
LFS_COUNT=0
while IFS=$'\t' read -r filename size sha256; do
  [[ -n "$filename" ]] || continue
  FILE_COUNT=$((FILE_COUNT + 1))
  TOTAL_BYTES=$((TOTAL_BYTES + size))
  if [[ -n "$sha256" ]]; then
    LFS_COUNT=$((LFS_COUNT + 1))
    printf '%s *%s\n' "$sha256" "$filename" >> "$CHECKSUMS"
  fi
  FILE_URL="$(python3 -c 'from urllib.parse import quote; import sys; print("https://huggingface.co/%s/resolve/main/%s" % (quote(sys.argv[1], safe="/"), quote(sys.argv[2], safe="/")))' "$REPO" "$filename")"
  printf '%s\n  out=%s\n' "$FILE_URL" "$filename" >> "$ARIA2_LIST"
done < "$MANIFEST"

if [[ "$FILE_COUNT" -eq 0 ]]; then
  printf 'Model repo %s contains no downloadable files.\n' "$REPO" >&2
  exit 1
fi

mkdir -p -- "$DIR"
FREE_KB_OVERRIDE="${HF_FREE_KB:-${HF_FREE_SPACE_KB:-${FETCH_MODEL_FREE_KB:-}}}"
if [[ -n "$FREE_KB_OVERRIDE" ]]; then
  FREE_KB="$FREE_KB_OVERRIDE"
else
  if ! FREE_KB="$(df -k "$DIR" | awk 'END { print $4 }')"; then
    printf 'Could not determine free disk space for %s.\n' "$DIR" >&2
    exit 1
  fi
fi
if [[ ! "$FREE_KB" =~ ^[0-9]+$ ]]; then
  printf 'Free-space value must be a nonnegative integer number of KiB; got %s.\n' "$FREE_KB" >&2
  exit 1
fi

RESERVE_KB=$((10 * 1024 * 1024))
DOWNLOAD_KB=$(((TOTAL_BYTES + 1023) / 1024))
REQUIRED_KB=$((DOWNLOAD_KB + RESERVE_KB))
FREE_BYTES=$((FREE_KB * 1024))
printf 'Files (%s):\n' "$FILE_COUNT"
while IFS=$'\t' read -r filename size sha256; do
  [[ -n "$filename" ]] || continue
  printf '  %s (%s bytes)\n' "$filename" "$size"
done < "$MANIFEST"
printf 'Total size: %s bytes\n' "$TOTAL_BYTES"
printf 'Free space: %s KiB (%s bytes)\n' "$FREE_KB" "$FREE_BYTES"

if [[ "$FREE_KB" -lt "$REQUIRED_KB" ]]; then
  printf 'Insufficient free space: need %s KiB for the download plus a 10 GiB reserve; have %s KiB.\n' \
    "$REQUIRED_KB" "$FREE_KB" >&2
  exit 1
fi

printf 'aria2c command: aria2c -i <generated-input-list> -d '
printf '%q' "$DIR"
printf ' -j 2 -x 8 -s 8 -k 64M --max-tries=0 --retry-wait=5 --continue=true --auto-file-renaming=false --console-log-level=warn\n'

if [[ "$DRY_RUN" == true ]]; then
  exit 0
fi

if ! command -v aria2c >/dev/null 2>&1; then
  printf 'aria2c is required to download model files; install it with: brew install aria2\n' >&2
  exit 1
fi

while IFS=$'\t' read -r filename size sha256; do
  [[ -n "$filename" ]] || continue
  if [[ "$filename" == */* ]]; then
    mkdir -p -- "$DIR/${filename%/*}"
  fi
done < "$MANIFEST"

aria2c -i "$ARIA2_LIST" -d "$DIR" -j 2 -x 8 -s 8 -k 64M \
  --max-tries=0 --retry-wait=5 --continue=true --auto-file-renaming=false --console-log-level=warn

if [[ "$LFS_COUNT" -gt 0 ]]; then
  if VERIFY_OUTPUT="$(cd "$DIR" && shasum -a 256 -c "$CHECKSUMS" 2>&1)"; then
    printf '%s\n' "$VERIFY_OUTPUT"
  else
    printf 'LFS SHA-256 verification failed:\n%s\n' "$VERIFY_OUTPUT" >&2
    exit 1
  fi
fi

printf 'Downloaded and verified %s file(s) from %s.\n' "$FILE_COUNT" "$REPO"
