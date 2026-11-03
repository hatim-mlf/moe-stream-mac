"""Resumable, verified Hugging Face model download (survives Wi-Fi drops and restarts).

  python tools/fetch_model.py mlx-community/Qwen3.5-122B-A10B-8bit [~/models]

Each file is fetched with `curl -C -` into <dir>/<name>.part, retried until complete, checked against
its published SHA-256 (LFS files) or size, then renamed. Re-running skips files that are already done.
"""
import hashlib
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HF = "https://huggingface.co"
RETRY_DELAY_S = 10
HASH_CHUNK = 16 * 1024 * 1024


def list_files(repo: str) -> list[dict]:
    with urllib.request.urlopen(f"{HF}/api/models/{repo}/tree/main?recursive=true", timeout=60) as resp:
        return [f for f in json.load(resp) if f["type"] == "file"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def is_valid(path: Path, meta: dict) -> bool:
    if not path.exists() or path.stat().st_size != meta["size"]:
        return False
    lfs = meta.get("lfs")
    return sha256(path) == lfs["oid"] if lfs else True


def fetch(repo: str, meta: dict, out_dir: Path) -> None:
    final = out_dir / meta["path"]
    if final.exists() and final.stat().st_size == meta["size"]:
        return
    part = final.with_name(final.name + ".part")
    part.parent.mkdir(parents=True, exist_ok=True)
    url = f"{HF}/{repo}/resolve/main/{meta['path']}"
    while not (part.exists() and part.stat().st_size >= meta["size"]):
        result = subprocess.run(["curl", "-sfL", "--retry", "20", "--retry-all-errors", "--retry-delay", "5",
                                 "--speed-limit", "10000", "--speed-time", "60", "-C", "-", "-o", str(part), url])
        if result.returncode not in (0, 33):  # 33: range error once a file is already complete
            print(f"{time.strftime('%T')} {meta['path']}: curl exit {result.returncode}, retrying", flush=True)
            time.sleep(RETRY_DELAY_S)
    if not is_valid(part, meta):
        part.unlink()
        raise RuntimeError(f"{meta['path']}: checksum mismatch, deleted the part file; re-run to fetch it again")
    part.rename(final)


def main() -> None:
    repo = sys.argv[1]
    out_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "~/models").expanduser() / repo.split("/")[-1]
    files = sorted(list_files(repo), key=lambda f: f["size"])  # small files first
    total = sum(f["size"] for f in files)
    print(f"{repo}: {len(files)} files, {total / 1e9:.1f} GB -> {out_dir}", flush=True)
    done = 0
    for meta in files:
        fetch(repo, meta, out_dir)
        done += meta["size"]
        print(f"{time.strftime('%T')} ok {meta['path']} ({done / total:.0%} of total)", flush=True)
    print("verifying all checksums...", flush=True)
    bad = [m["path"] for m in files if not is_valid(out_dir / m["path"], m)]
    if bad:
        raise SystemExit(f"FAILED checksum: {bad}")
    print(f"MODEL READY: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
