# Brief 47: resumable, verified model download script (Codex)

<task>
Write `scripts/fetch-model.sh REPO DIR [--dry-run]` (bash 3.2 compatible: macOS /bin/bash, `set -euo pipefail`,
no empty-array expansion under `set -u`) that downloads every file of a public Hugging Face model repo into DIR:
1. Get the file list and each LFS file's sha256 from `https://huggingface.co/api/models/REPO?blobs=true`
   (parse with `python3 -c`, using `json.loads(..., strict=False)`: these responses can contain control
   characters). Refuse with a clear message if the repo is not found or is gated.
2. Write an aria2c input file (url = `https://huggingface.co/REPO/resolve/main/<file>`, `out=<file>`), then run
   `aria2c -i <list> -d DIR -j 2 -x 8 -s 8 -k 64M --max-tries=0 --retry-wait=5 --continue=true
   --auto-file-renaming=false --console-log-level=warn` (resumable: rerunning continues). Refuse if aria2c is
   not installed (print `brew install aria2`).
3. Verify every LFS file with `shasum -a 256 -c` against the Hub hashes; exit non-zero listing mismatches.
4. Before downloading, compare the total size with free disk space (`df -k DIR`) and refuse if less than size +
   10 GB would remain.
5. `--dry-run` prints the file list, total size, free space and the aria2c command without downloading.
Also update `docs/MODELS.md` only if you find an error in it (do not reformat).
</task>

<tests>
`bash -n scripts/fetch-model.sh`; `tools/test_fetch_model.py` (pytest) runs the script with `--dry-run`
against a fake API response (make the API base URL overridable by env `HF_API_BASE` for the test, default
`https://huggingface.co`) and checks the printed list, size and the refusal when disk space is short (make the
free-space value overridable by env for the test). You have no network in the sandbox: do not try real
downloads.
</tests>

<constraints>
Do not modify other files. No commit. Report files and test output.
</constraints>
