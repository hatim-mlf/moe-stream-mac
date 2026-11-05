# Brief 44: start mem_guard in every MLX entry point (Codex)

<task>
`tools/mem_guard.py` stops a process before macOS runs out of wired memory (a kernel panic happened on
2026-11-05 without it). Today only some entry points start it. Add it to every *runnable* script under tools/
that imports MLX and can allocate model-sized memory: at the top of its `main()` (or the `if __name__ ==
"__main__":` block), before any model or large array is created, call `mem_guard.start()` and print the same
one-line limit message that tools/ft_lora_lowmem.py prints. Scripts to cover (check each is runnable; skip
pure libraries that are only imported): ft_lora_probe.py, ft_night.py, gdn_vjp_mem.py, route_nll.py,
sweep_router_nll.py, steering.py, router_bias.py, gen_gdn_fixture.py, gen_ops_fixture.py, gen_qmat_fixture.py,
and any other runnable MLX script you find with `grep -l "mlx" tools/*.py`. Do not start it at import time in
libraries (frozen_gdn, gdn_checkpoint, ft_lowmem_loss, gdn_vjp, mem_budget): those are imported by guarded
entry points. Also add `tools/test_mem_guard_entry_points.py`: static check (ast) that every runnable MLX
script in tools/ calls mem_guard.start() in its main path, with an explicit allow-list for libraries.
</task>

<constraints>
Do not change any script's other behaviour. Named constants, type hints on new code, no silent excepts. No
commit. Report the list of files changed and the allow-list.
</constraints>
