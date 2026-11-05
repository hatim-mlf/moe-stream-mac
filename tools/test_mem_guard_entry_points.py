"""Static checks for wired-memory guards on runnable MLX entry points."""

from __future__ import annotations

import ast
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parent
LIBRARY_ALLOWLIST: frozenset[str] = frozenset(
    {
        "frozen_gdn.py",
        "gdn_checkpoint.py",
        "ft_lowmem_loss.py",
        "gdn_vjp.py",
        "mem_budget.py",
        "mem_preflight.py",  # tokenizer + budget arithmetic only; never loads weights
        "mem_trainer.py",  # library for mem_profile trainer-step, which starts the guard
    }
)
MLX_SUPERVISOR_SCRIPTS: frozenset[str] = frozenset({"ft_night.py"})


def _imports_mlx(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(
                alias.name == "mlx"
                or alias.name.startswith("mlx.")
                or alias.name == "mlx_lm"
                or alias.name.startswith("mlx_lm.")
                for alias in node.names
            ):
                return True
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            if (
                node.module == "mlx"
                or node.module.startswith("mlx.")
                or node.module == "mlx_lm"
                or node.module.startswith("mlx_lm.")
            ):
                return True
    return False


def _is_main_guard(node: ast.If) -> bool:
    test = node.test
    if not isinstance(test, ast.Compare) or len(test.ops) != 1:
        return False
    if not isinstance(test.ops[0], ast.Eq):
        return False
    left, = test.comparators
    return (
        isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
        and isinstance(left, ast.Constant)
        and left.value == "__main__"
    ) or (
        isinstance(test.left, ast.Constant)
        and test.left.value == "__main__"
        and isinstance(left, ast.Name)
        and left.id == "__name__"
    )


def _calls_mem_guard_start(nodes: list[ast.AST]) -> bool:
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "start"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "mem_guard"
        for root in nodes
        for node in ast.walk(root)
    )


def _local_calls(nodes: list[ast.AST]) -> set[str]:
    return {
        node.func.id
        for root in nodes
        for node in ast.walk(root)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }


def _entrypoint_reaches_guard(tree: ast.Module) -> bool:
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    main_guards = [
        node
        for node in tree.body
        if isinstance(node, ast.If) and _is_main_guard(node)
    ]
    if main_guards:
        entry_nodes = [statement for guard in main_guards for statement in guard.body]
    else:
        entry_nodes = [
            node
            for node in tree.body
            if not isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                    ast.ClassDef,
                    ast.Import,
                    ast.ImportFrom,
                ),
            )
        ]

    if _calls_mem_guard_start(entry_nodes):
        return True

    pending = list(_local_calls(entry_nodes))
    visited: set[str] = set()
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        function = functions.get(name)
        if function is None:
            continue
        body = list(function.body)
        if _calls_mem_guard_start(body):
            return True
        pending.extend(_local_calls(body) - visited)
    return False


def test_runnable_mlx_entry_points_start_mem_guard() -> None:
    for path in sorted(TOOLS_DIR.glob("*.py")):
        if path.name.startswith("test_") or path.name in LIBRARY_ALLOWLIST:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if not _imports_mlx(tree) and path.name not in MLX_SUPERVISOR_SCRIPTS:
            continue
        assert _entrypoint_reaches_guard(tree), (
            f"{path.name} is an MLX entry point without mem_guard.start() "
            "reachable from its main path"
        )
