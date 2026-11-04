# Task: write engine/hybrid/expert_reader.py (Python 3.14, numpy, standard library only)

Write a module with ONE class, `ExpertReader`, that reads selected MoE experts straight from an MLX
safetensors checkpoint on disk, in parallel, without loading whole tensors.

## Safetensors format
Each `*.safetensors` file: 8 bytes little-endian uint64 N, then N bytes of JSON header, then the data.
Header: {"tensor name": {"dtype": "U32"|"BF16"|..., "shape": [...], "data_offsets": [start, end]}, ...,
optionally "__metadata__": {...} (skip it). Absolute file offset of a tensor = 8 + N + start.
`model.safetensors.index.json` in the model dir has {"weight_map": {"tensor name": "file name", ...}}.

## Expert tensors
For layer L and projection P in ("gate_proj", "up_proj", "down_proj") the three tensors are
`language_model.model.layers.{L}.mlp.switch_mlp.{P}.weight` (dtype U32), `.scales` (BF16), `.biases` (BF16).
Each has shape [E, rows, cols_part]; expert e is ONE contiguous byte range: offset + e * (nbytes / E).

## API (exact)
```python
class ExpertReader:
    def __init__(self, model_dir: Path, threads: int = 8):
        """Parse the index and every needed header ONCE. Keep a dict:
        (layer, proj, part) -> (file_path, abs_offset, per_expert_nbytes, per_expert_shape, numpy dtype)
        with numpy dtype np.uint32 for "U32" and np.uint16 for "BF16" (bf16 kept as raw uint16 bits).
        Open each shard file ONCE with os.open(path, os.O_RDONLY) and keep the fds.
        Set self.n_layers = number of distinct layers that have switch_mlp tensors.
        Create a concurrent.futures.ThreadPoolExecutor(max_workers=threads) kept for the object's lifetime."""

    def read(self, layer: int, proj: str, experts: np.ndarray) -> dict:
        """Return {"weight": arr, "scales": arr, "biases": arr}, each of shape [len(experts), *per_expert_shape]
        and the stored dtype, holding experts in the given order (duplicates allowed).
        Allocate each output with np.empty, then submit one task per (part, expert) to the pool; each task does
        os.preadv(fd, [memoryview(out[part][i]).cast("B")], offset + e * per_expert_nbytes) and checks it read
        exactly per_expert_nbytes bytes (raise OSError otherwise). Wait for all tasks; re-raise any error.
        Raise KeyError with a clear message if (layer, proj) is unknown."""

    def close(self) -> None:
        """Shut down the pool and close every fd."""
```

Rules: imports limited to json, os, struct, concurrent.futures, pathlib, numpy. Every method under 40 lines
(use small private helpers). No global state. Return only the complete file in one fenced python block.


## Your previous attempt failed

Test output:
```
    raise KeyError(f"Tensor (layer={layer}, proj={proj}) not found in index.")
KeyError: 'Tensor (layer=0, proj=down_proj) not found in index.'
```

Code review (rewrite the file):
1. "part" means the LAST component of the tensor name: "weight", "scales" or "biases". It is NOT a split tensor.
   The mapping key is (layer, proj, part), e.g. (0, "down_proj", "scales"). read() returns one array per part.
2. Parse names with exactly this regex (and skip names that do not match):
     NAME = re.compile(r"layers\.(\d+)\.mlp\.switch_mlp\.(gate_proj|up_proj|down_proj)\.(weight|scales|biases)$")
     m = NAME.search(tensor_name); layer, proj, part = int(m.group(1)), m.group(2), m.group(3)
   (You may add `import re`.) parts[-4] is "mlp", not the layer number.
3. per_expert_shape = tuple(shape[1:]); per_expert_nbytes = (end - start) // shape[0].
4. Parse each shard's header ONCE (cache by file), not once per tensor.
5. Delete every "thinking aloud" comment ("Let's assume...", "Actually...", "The prompt says..."). Comments only
   where the code is not obvious.


Fix it. Reply with the complete corrected file.