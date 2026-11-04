# Review notes for tools/router_bias.py (apply every note exactly; change nothing else)

1. `sum_gradients` is used but never imported: add `from mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock, sum_gradients`
   if qwen3_next exports it, otherwise import it from where qwen3_next imports it (check with
   `grep -n sum_gradients` in the mlx_lm package).
2. `find_moe_blocks` must raise ValueError("no Qwen3NextSparseMoeBlock found") when the list is empty.
3. `coding_set` tie order is wrong: `np.argsort(...)[::-1]` puts the HIGHER id first on ties. Use
   `np.argsort(-log_ratios, kind="stable")[:n_per_layer]` so ties pick the lower expert id.
4. Replace the closure inside `serve()` with a module-level function
   `patch_provider(provider_cls, bias: np.ndarray, mode: str) -> None` that wraps `provider_cls.load`.
   `load` returns a `(model, tokenizer)` TUPLE: install the bias on `result[0]`, and only when that model object
   has not been patched before (remember `id(model)` in a set inside patch_provider), then return the result unchanged.
5. Replace `serve()` with `main(argv=None) -> int`: argparse with a required positional `command` (choices: "serve"),
   `--bias` (required path), `--mode` (choices MODES, default "select"), and `parse_known_args` so all other args go
   to the server. It loads the bias, calls `patch_provider(mlx_lm.server.ModelProvider, bias, mode)`, sets
   `sys.argv = ["mlx_lm.server", *rest]`, and calls `mlx_lm.server.main()`; return 0.
   End the file with `if __name__ == "__main__": raise SystemExit(main())`.

Return the complete corrected file in one fenced python block.
