"""Tests for tools/polyglot_bench.py (Aider Polyglot exercise loading, prompting, reply parsing, test running)."""
import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import polyglot_bench as pb  # noqa: E402

FENCE = "`" * 3


def make_python_exercise(root: Path) -> Path:
    ex = root / "python" / "exercises" / "practice" / "two-fer"
    (ex / ".docs").mkdir(parents=True)
    (ex / ".meta").mkdir()
    (ex / ".docs" / "instructions.md").write_text("# Two-fer\n\nReturn 'One for X, one for me.'\n")
    (ex / ".docs" / "instructions.append.md").write_text("Default X is 'you'.\n")
    (ex / ".meta" / "config.json").write_text(json.dumps({"files": {
        "solution": ["two_fer.py"], "test": ["two_fer_test.py"], "example": [".meta/example.py"]}}))
    (ex / ".meta" / "example.py").write_text("SECRET_REFERENCE = 1\n")
    (ex / "two_fer.py").write_text("def two_fer(name='you'):\n    pass\n")
    (ex / "two_fer_test.py").write_text(
        "import unittest\nfrom two_fer import two_fer\n\n"
        "class T(unittest.TestCase):\n"
        "    def test_default(self):\n        self.assertEqual(two_fer(), 'One for you, one for me.')\n"
        "    def test_name(self):\n        self.assertEqual(two_fer('Al'), 'One for Al, one for me.')\n")
    return ex


GOOD_PY = "def two_fer(name='you'):\n    return f'One for {name}, one for me.'\n"


def make_cpp_exercise(root: Path) -> Path:
    ex = root / "cpp" / "exercises" / "practice" / "add-one"
    (ex / ".docs").mkdir(parents=True)
    (ex / ".meta").mkdir()
    (ex / ".docs" / "instructions.md").write_text("Implement add_one(int).\n")
    (ex / ".meta" / "config.json").write_text(json.dumps({"files": {
        "solution": ["add_one.cpp", "add_one.h"], "test": ["add_one_test.cpp"],
        "example": [".meta/example.cpp", ".meta/example.h"]}}))
    (ex / "add_one.h").write_text("#pragma once\nint add_one(int x);\n")
    (ex / "add_one.cpp").write_text('#include "add_one.h"\n')
    (ex / "add_one_test.cpp").write_text(
        '#include "add_one.h"\n#include <cstdlib>\nint main() { return add_one(1) == 2 ? 0 : 1; }\n')
    (ex / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.5.1)\nproject(add_one CXX)\n"
        "add_executable(add_one add_one_test.cpp add_one.cpp)\n"
        "add_custom_command(TARGET add_one POST_BUILD COMMAND add_one)\n")
    return ex


@pytest.fixture
def py_root(tmp_path):
    make_python_exercise(tmp_path / "bench")
    return tmp_path / "bench"


def test_list_exercises_sorted_by_language_then_name(py_root):
    make_cpp_exercise(py_root)
    assert pb.list_exercises(py_root, ("python", "cpp")) == [("python", "two-fer"), ("cpp", "add-one")]


def test_load_exercise_reads_instructions_and_stubs(py_root):
    ex = pb.load_exercise(py_root, "python", "two-fer")
    assert ex["lang"] == "python" and ex["name"] == "two-fer"
    assert "Two-fer" in ex["instructions"] and "Default X is 'you'." in ex["instructions"]
    assert ex["solution_files"] == {"two_fer.py": "def two_fer(name='you'):\n    pass\n"}
    assert ex["test_files"] == ["two_fer_test.py"]


def test_prompt_names_every_solution_file_and_hides_reference(py_root):
    prompt = pb.build_prompt(pb.load_exercise(py_root, "python", "two-fer"))
    assert "Two-fer" in prompt and "two_fer.py" in prompt and "pass" in prompt
    assert "SECRET_REFERENCE" not in prompt
    assert "two_fer_test.py" not in prompt  # tests stay hidden, as in Aider's benchmark


def test_parse_reply_matches_file_names_before_blocks():
    reply = (f"Here you go.\n\n**add_one.h**\n{FENCE}cpp\n#pragma once\nint add_one(int x);\n{FENCE}\n\n"
             f"`add_one.cpp`\n{FENCE}cpp\nint add_one(int x) {{ return x + 1; }}\n{FENCE}\n")
    files = pb.parse_reply(reply, ["add_one.cpp", "add_one.h"])
    assert files == {"add_one.h": "#pragma once\nint add_one(int x);\n",
                     "add_one.cpp": "int add_one(int x) { return x + 1; }\n"}


def test_parse_reply_single_unnamed_block_goes_to_single_file():
    files = pb.parse_reply(f"{FENCE}python\n{GOOD_PY}{FENCE}", ["two_fer.py"])
    assert files == {"two_fer.py": GOOD_PY}


def test_parse_reply_without_blocks_returns_empty():
    assert pb.parse_reply("I cannot do that.", ["two_fer.py"]) == {}


def test_prepare_workdir_excludes_meta_and_keeps_tests(py_root, tmp_path):
    ex = pb.load_exercise(py_root, "python", "two-fer")
    work = pb.prepare_workdir(ex, tmp_path / "work")
    assert work.name == "two-fer"
    assert (work / "two_fer.py").exists() and (work / "two_fer_test.py").exists()
    assert not (work / ".meta").exists() and not (work / ".docs").exists()


def test_python_tests_pass_and_fail(py_root, tmp_path):
    ex = pb.load_exercise(py_root, "python", "two-fer")
    work = pb.prepare_workdir(ex, tmp_path / "work")
    passed, output = pb.run_tests(work, ex)
    assert passed is False and output
    pb.write_files(work, {"two_fer.py": GOOD_PY})
    passed, _ = pb.run_tests(work, ex)
    assert passed is True


def test_write_files_refuses_names_outside_solution_files(py_root, tmp_path):
    ex = pb.load_exercise(py_root, "python", "two-fer")
    work = pb.prepare_workdir(ex, tmp_path / "work")
    with pytest.raises(ValueError):
        pb.write_files(work, {"two_fer_test.py": "x = 1\n"}, allowed=ex["solution_files"])
    with pytest.raises(ValueError):
        pb.write_files(work, {"../escape.py": "x = 1\n"}, allowed=ex["solution_files"])


@pytest.mark.skipif(shutil.which("cmake") is None, reason="cmake not installed")
def test_cpp_build_runs_tests(tmp_path):
    root = tmp_path / "bench"
    make_cpp_exercise(root)
    ex = pb.load_exercise(root, "cpp", "add-one")
    work = pb.prepare_workdir(ex, tmp_path / "work")
    passed, output = pb.run_tests(work, ex)
    assert passed is False and output  # stub has no definition: link error
    pb.write_files(work, {"add_one.cpp": '#include "add_one.h"\nint add_one(int x) { return x + 1; }\n'})
    passed, _ = pb.run_tests(work, ex)
    assert passed is True


def test_test_output_is_clipped_to_tail(py_root, tmp_path, monkeypatch):
    monkeypatch.setattr(pb, "MAX_TEST_OUTPUT_CHARS", 200)
    ex = pb.load_exercise(py_root, "python", "two-fer")
    work = pb.prepare_workdir(ex, tmp_path / "work")
    _, output = pb.run_tests(work, ex)
    assert len(output) <= 200


def test_timeout_reports_failure(py_root, tmp_path, monkeypatch):
    monkeypatch.setattr(pb, "TEST_TIMEOUT_SECONDS", 1)
    ex = pb.load_exercise(py_root, "python", "two-fer")
    work = pb.prepare_workdir(ex, tmp_path / "work")
    pb.write_files(work, {"two_fer.py": "import time\ntime.sleep(5)\n" + GOOD_PY})
    passed, output = pb.run_tests(work, ex)
    assert passed is False and "timed out" in output.lower()
