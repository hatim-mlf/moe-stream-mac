"""Tests for the resumable Hugging Face model fetch script."""

import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "fetch-model.sh"
FREE_SPACE_REQUIRED_FOR_TEST = 10 * 1024 * 1024 + 1


def write_api_response(path: Path) -> Path:
    payload = {
        # Exercise the non-strict JSON parsing used for some Hub responses.
        "description": "control\u0001character",
        "siblings": [
            {"rfilename": "config.json", "size": 13},
            {
                "rfilename": "weights/model.safetensors",
                "size": 8,
                "lfs": {"sha256": "a" * 64, "size": 8},
            },
        ],
    }
    encoded = json.dumps(payload).replace(r"\u0001", "\x01")
    path.write_text(encoded, encoding="utf-8")
    return path


def run_dry_run(api_file: Path, output_dir: Path, free_kb: int) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.update(
        {
            "HF_API_FILE": str(api_file),
            "HF_FREE_KB": str(free_kb),
        }
    )
    return subprocess.run(
        [str(SCRIPT), "test-org/test-model", str(output_dir), "--dry-run"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )


def test_dry_run_prints_files_total_and_aria2_command(tmp_path: Path) -> None:
    api_file = write_api_response(tmp_path / "model.json")
    result = run_dry_run(
        api_file,
        tmp_path / "models",
        FREE_SPACE_REQUIRED_FOR_TEST,
    )

    assert result.returncode == 0, result.stderr
    assert "config.json (13 bytes)" in result.stdout
    assert "weights/model.safetensors (8 bytes)" in result.stdout
    assert "Total size: 21 bytes" in result.stdout
    assert "aria2c command:" in result.stdout


def test_dry_run_refuses_when_free_space_is_short(tmp_path: Path) -> None:
    api_file = write_api_response(tmp_path / "model.json")
    result = run_dry_run(api_file, tmp_path / "models", free_kb=1)

    assert result.returncode != 0
    assert "Insufficient free space" in result.stderr
    assert "have 1 KiB" in result.stderr
