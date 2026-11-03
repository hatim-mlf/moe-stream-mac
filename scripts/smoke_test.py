"""Phase 0 smoke test: receiver alone vs C2C on a few prompts, with wall-clock timing.

Usage: python scripts/smoke_test.py [--device cpu|cuda] [--max-new-tokens 128]
Writes results/<timestamp>-smoke/metrics.json.
"""

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parent.parent
UPSTREAM_DIR = ROOT / "third_party" / "C2C"
sys.path.insert(0, str(UPSTREAM_DIR))

from script.playground.inference_example import load_rosetta_model  # noqa: E402

RECEIVER = "Qwen/Qwen3-0.6B"
SHARER = "Qwen/Qwen2.5-0.5B-Instruct"
FUSER_REPO = "nics-efc/C2C_Fuser"
FUSER_SUBDIR = "qwen3_0.6b+qwen2.5_0.5b_Fuser"
PROMPTS = (
    "Say hello in one short sentence.",
    "What is 17 * 23? Answer with the number only.",
    "Which planet is known as the Red Planet? Answer in one word.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser.parse_args()


def upstream_sha() -> str:
    result = subprocess.run(
        ["git", "-C", str(UPSTREAM_DIR), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def encode(tokenizer, prompt: str, device: torch.device):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    return tokenizer(text, return_tensors="pt").to(device)


def c2c_kv_index(seq_len: int, device: torch.device) -> list:
    # [1, 0] = fuse sharer cache into the prompt tokens; [-1, 0] = no fusion for the response.
    instruction = torch.tensor([1, 0], dtype=torch.long).repeat(seq_len - 1, 1).unsqueeze(0)
    response = torch.tensor([[-1, 0]], dtype=torch.long).unsqueeze(0)
    return [instruction.to(device), response.to(device)]


def timed_generate(generate_fn) -> tuple:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        outputs = generate_fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return outputs, time.perf_counter() - start


def run_prompt(prompt, tokenizer, receiver, c2c_model, device, gen_kwargs) -> dict:
    inputs = encode(tokenizer, prompt, device)
    prompt_len = inputs["input_ids"].shape[1]

    base_out, base_s = timed_generate(lambda: receiver.generate(**inputs, **gen_kwargs))
    c2c_out, c2c_s = timed_generate(
        lambda: c2c_model.generate(**inputs, kv_cache_index=c2c_kv_index(prompt_len, device), **gen_kwargs)
    )
    return {
        "prompt": prompt,
        "receiver_alone": {"seconds": round(base_s, 3),
                           "text": tokenizer.decode(base_out[0, prompt_len:], skip_special_tokens=True)},
        "c2c": {"seconds": round(c2c_s, 3),
                "text": tokenizer.decode(c2c_out[0, prompt_len:], skip_special_tokens=True)},
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if not UPSTREAM_DIR.exists():
        sys.exit("third_party/C2C missing. Run: bash scripts/setup.sh")

    ckpt_root = snapshot_download(repo_id=FUSER_REPO, allow_patterns=[f"{FUSER_SUBDIR}/*"])
    model_config = {"rosetta_config": {
        "base_model": RECEIVER, "teacher_model": SHARER,
        "checkpoints_dir": f"{ckpt_root}/{FUSER_SUBDIR}/final",
    }}
    c2c_model, tokenizer = load_rosetta_model(model_config, eval_config={}, device=device)
    receiver = AutoModelForCausalLM.from_pretrained(
        RECEIVER, torch_dtype=torch.bfloat16, device_map={"": device}
    ).eval()

    gen_kwargs = {"do_sample": False, "max_new_tokens": args.max_new_tokens}
    rows = [run_prompt(p, tokenizer, receiver, c2c_model, device, gen_kwargs) for p in PROMPTS]

    out_dir = ROOT / "results" / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-smoke"
    out_dir.mkdir(parents=True)
    report = {
        "upstream_sha": upstream_sha(), "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "python": platform.python_version(), "torch": torch.__version__,
        "receiver": RECEIVER, "sharer": SHARER, "fuser": FUSER_SUBDIR,
        "gen_kwargs": gen_kwargs, "rows": rows,
    }
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"\nSaved {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
