"""Aider Polyglot benchmark: load exercises, build prompts, parse replies, run tests."""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

MAX_TEST_OUTPUT_CHARS = 4000
TEST_TIMEOUT_SECONDS = 180
LANGS: tuple[str, ...] = ("python", "cpp")


def list_exercises(root: Path, langs: tuple[str, ...] = LANGS) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    for lang in langs:
        practice_dir = root / lang / "exercises" / "practice"
        if not practice_dir.is_dir():
            continue
        for entry in sorted(practice_dir.iterdir()):
            if entry.is_dir():
                result.append((lang, entry.name))
    return result


def load_exercise(root: Path, lang: str, name: str) -> dict[str, Any]:
    ex_dir = root / lang / "exercises" / "practice" / name
    config_path = ex_dir / ".meta" / "config.json"
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    files_cfg = config["files"]
    solution_names: list[str] = files_cfg.get("solution", [])
    test_names: list[str] = files_cfg.get("test", [])

    docs_dir = ex_dir / ".docs"
    parts: list[str] = []
    for fname in ("introduction.md", "instructions.md", "instructions.append.md"):
        p = docs_dir / fname
        if p.is_file():
            parts.append(p.read_text(encoding="utf-8").rstrip())
    instructions = "\n\n".join(parts)

    solution_files: dict[str, str] = {}
    for sname in solution_names:
        sp = ex_dir / sname
        if sp.is_file():
            solution_files[sname] = sp.read_text(encoding="utf-8")

    return {
        "lang": lang,
        "name": name,
        "dir": ex_dir,
        "instructions": instructions,
        "solution_files": solution_files,
        "test_files": test_names,
    }


def build_prompt(ex: dict[str, Any]) -> str:
    instructions: str = ex["instructions"]
    solution_files: dict[str, str] = ex["solution_files"]
    names = list(solution_files.keys())
    comma_names = ", ".join(names)
    paragraph = (
        f"Use the above instructions to modify the supplied files: {comma_names}. "
        "Keep and implement the existing function and class names; they will be called from unit tests. "
        "Only use the standard library. Reply with the complete new content of every file you change: "
        "the file name on its own line, followed by a fenced code block."
    )
    parts: list[str] = [instructions, paragraph]
    for name in names:
        content = solution_files[name]
        body = content if content.endswith("\n") else content + "\n"
        parts.append(f"{name}\n```\n{body}```")
    return "\n\n".join(parts)


def parse_reply(reply: str, filenames: list[str]) -> dict[str, str]:
    lines = reply.split("\n")
    n = len(lines)
    blocks: list[tuple[int, str]] = []  # (opening_fence_line_index, content)
    i = 0
    while i < n:
        stripped = lines[i].strip()
        if stripped.startswith("```"):
            opening_idx = i
            i += 1
            content_lines: list[str] = []
            while i < n:
                if lines[i].strip() == "```":
                    i += 1
                    break
                content_lines.append(lines[i])
                i += 1
            content = "\n".join(content_lines)
            if not content.endswith("\n"):
                content += "\n"
            blocks.append((opening_idx, content))
        else:
            i += 1

    if not blocks:
        return {}

    result: dict[str, str] = {}
    for opening_idx, content in blocks:
        j = opening_idx - 1
        while j >= 0:
            prev = lines[j].strip()
            if prev:
                for fname in filenames:
                    if fname in prev:
                        result[fname] = content
                        break
                break
            j -= 1

    if not result and len(filenames) == 1:
        result[filenames[0]] = blocks[-1][1]

    return result


def prepare_workdir(ex: dict[str, Any], dest_parent: Path) -> Path:
    name: str = ex["name"]
    src: Path = ex["dir"]
    dest = dest_parent / name
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for entry in src.iterdir():
        if entry.name.startswith("."):
            continue
        dest_entry = dest / entry.name
        if entry.is_dir():
            shutil.copytree(entry, dest_entry)
        else:
            shutil.copy2(entry, dest_entry)
    return dest


def write_files(work: Path, files: dict[str, str], allowed: Any = None) -> None:
    allowed_set: set[str] | None = None
    if allowed is not None:
        allowed_set = set(allowed)
    for name, content in files.items():
        if allowed_set is not None and name not in allowed_set:
            raise ValueError(f"File name {name!r} not in allowed set")
        if "/" in name or "\\" in name or name == "..":
            raise ValueError(f"Invalid file name: {name!r}")
        path = work / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _run(cmd: list[str] | str, cwd: Path, timeout: float) -> tuple[int, str] | None:
    """Run in a new process group; on timeout kill the whole group (make + test binary) and return None."""
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            shell=isinstance(cmd, str), start_new_session=True)
    try:
        output, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate()
        return None
    return proc.returncode, output


def run_tests(work: Path, ex: dict[str, Any]) -> tuple[bool, str]:
    lang: str = ex["lang"]
    test_files: list[str] = ex["test_files"]
    timeout = TEST_TIMEOUT_SECONDS
    max_chars = MAX_TEST_OUTPUT_CHARS

    if lang == "python":
        cmd: list[str] | str = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *test_files]
        cwd = work
    else:  # cpp
        cwd = work / "build"
        if cwd.exists():
            shutil.rmtree(cwd)
        cwd.mkdir(parents=True)
        cmd = "cmake -Wno-dev -DEXERCISM_RUN_ALL_TESTS=1 .. && make"
    result = _run(cmd, cwd, timeout)
    if result is None:
        return (False, f"tests timed out after {timeout} s")
    returncode, output = result
    return (returncode == 0, output[-max_chars:])
