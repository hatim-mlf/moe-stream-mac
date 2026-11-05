# Brief 35: system wired-memory guard for training runs

<task>
On 2026-11-05 13:45 the Mac (M1 Max, 32 GiB) kernel-panicked during an `mlx_lm.lora` probe: with
`iogpu.wired_limit_mb` at ~31 GiB the training process wired 31.26 GiB, macOS was left ~14 MB free, `watchdogd`
missed check-ins for 94 s and the kernel panicked (`watchdog timeout`). The user keeps the high wired limit, so the
protection has to live in the training process. `mx.set_memory_limit` is soft (verified: a 3 GiB allocation under a
1 GiB limit succeeds), so the guard watches the **system** wired page count and kills the process before the OS
starves.

Write `tools/mem_guard.py` (stdlib only) with exactly this interface, as used by `tools/test_mem_guard.py`
(already written by the lead; read it, do not change it):

- `DEFAULT_HEADROOM_GB = 2.5`, `MIN_HEADROOM_GB = 1.0`, `POLL_SECONDS = 0.2`, `EXIT_CODE = 86`.
- `parse_vm_stat(text: str) -> int`: wired bytes = "Pages wired down" count × the page size from the header
  line `(page size of N bytes)`. Raise `ValueError` if either is missing or unparsable (never default the page size).
- `read_wired_bytes() -> int`: run `/usr/bin/vm_stat`, parse it. Errors propagate.
- `total_memory_bytes() -> int`: `sysctl -n hw.memsize`.
- `headroom_bytes_from_env(env: Mapping[str, str]) -> int`: `MEM_GUARD_HEADROOM_GB` (float GiB), default
  `DEFAULT_HEADROOM_GB`; `ValueError` for non-numeric, non-finite or `< MIN_HEADROOM_GB`.
- `is_breach(wired: int, total: int, headroom: int) -> bool`: `wired >= total - headroom`.
- `start(read_wired=read_wired_bytes, total_bytes=None, headroom_bytes=None, poll_seconds=POLL_SECONDS,
  on_breach=abort) -> threading.Thread`: `None` means `total_memory_bytes()` / `headroom_bytes_from_env(os.environ)`.
  Check once synchronously before starting the thread (fire immediately if already over). Then a daemon thread polls
  every `poll_seconds`. On breach call `on_breach(message)` **once** and stop polling. The message names wired GiB
  and limit GiB formatted `%.2f` (e.g. "wired 31.00 GiB >= limit 30.00 GiB (total 32.00, headroom 2.00)") and says
  the run was stopped to avoid a kernel watchdog panic. If `read_wired` raises, that is also a breach (fail safe):
  call `on_breach` once with a message containing the exception text. Never swallow it silently.
- `abort(message: str) -> None`: print `mem_guard: <message>` to stderr, flush stdout/stderr, `os._exit(EXIT_CODE)`
  (from a thread; `sys.exit` would not stop the process).

Then `tools/ft_lora_lowmem.py` (exists, has an uncommitted GDN_CHECKPOINT change; keep it): `import mem_guard`
and call `mem_guard.start()` right before `lora.main()`, and print one line at startup:
`mem_guard: limit <limit GiB> GiB (total <total>, headroom <headroom>)`. Do not change its other behaviour.
</task>

<tests>
`pytest tools/test_mem_guard.py` (run from the repo root with `tools` on `sys.path` like the other tool tests;
the last test runs the real `vm_stat`/`sysctl`, which works on macOS).
</tests>

<constraints>
- Do not modify `.venv`, the test file, or any other file. Named constants, type hints, functions < 50 lines,
  no silent excepts.
- Do not commit. Report: files changed, test result, anything unverified.
</constraints>
