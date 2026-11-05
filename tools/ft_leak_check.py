#!/usr/bin/env python3
"""Reject heldout-v2 task ids or public entry points outside their frozen suite."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tools" / "ft_eval_heldout_v2.json"
SUITE_ROOT = (ROOT / "evals" / "heldout-v2").resolve()


def main():
    manifest = json.loads(MANIFEST.read_text())
    needles = sorted({task["id"] for task in manifest["tasks"]} |
                     {name for task in manifest["tasks"] for name in task["public_names"]})
    patterns = [(value, re.compile(r"(?<![A-Za-z0-9_])" + re.escape(value) + r"(?![A-Za-z0-9_])"))
                for value in needles]
    roots = [ROOT / "data" / "finetune", ROOT / "evals"]
    failures = []
    for scan_root in roots:
        if not scan_root.exists():
            continue
        for path in sorted(p for p in scan_root.rglob("*") if p.is_file()):
            resolved = path.resolve()
            if scan_root.name == "evals" and (resolved == SUITE_ROOT or SUITE_ROOT in resolved.parents):
                continue
            try:
                content = path.read_text(errors="replace")
            except OSError:
                continue
            for line_no, line in enumerate(content.splitlines(), 1):
                for needle, pattern in patterns:
                    if pattern.search(line):
                        failures.append(f"{path}:{line_no}: heldout-v2 leak token {needle!r}")
    if failures:
        print("\n".join(failures))
        return 1
    print(f"leak check passed: {len(needles)} heldout-v2 ids/public names absent outside suite")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
