```python
import json
import os
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


class ExpertReader:
    """Reads selected MoE experts from MLX safetensors files in parallel."""

    DTYPE_MAP: Dict[str, np.dtype] = {
        "U32": np.uint32,
        "BF16": np.uint16,
    }

    def __init__(self, model_dir: Path, threads: int = 8) -> None:
        self._model_dir = model_dir
        self._index: Dict[str, str] = self._load_index()
        self._tensors: Dict[Tuple[int, str, int], Dict[str, Any]] = {}
        self._fds: Dict[str, int] = {}
        self._layers: set[int] = set()
        self._build_mapping()
        self.n_layers = len(self._layers)
        self._pool = ThreadPoolExecutor(max_workers=threads)

    def _load_index(self) -> Dict[str, str]:
        index_path = self._model_dir / "model.safetensors.index.json"
        with open(index_path, "r") as f:
            data = json.load(f)
        return data.get("weight_map", {})

    def _build_mapping(self) -> None:
        for tensor_name, file_name in self._index.items():
            if "switch_mlp" not in tensor_name:
                continue
            parts = tensor_name.split(".")
            try:
                layer_idx = int(parts[-4])
            except (ValueError, IndexError):
                continue
            proj = parts[-2]
            if proj not in ("gate_proj", "up_proj", "down_proj"):
                continue
            self._layers.add(layer_idx)
            shard_path = self._model_dir / file_name
            self._open_shard(shard_path)
            self._parse_shard_header(shard_path, tensor_name, layer_idx, proj)

    def _open_shard(self, path: Path) -> None:
        if str(path) not in self._fds:
            self._fds[str(path)] = os.open(str(path), os.O_RDONLY)

    def _parse_shard_header(self, path: Path, tensor_name: str, layer: int, proj: str) -> None:
        fd = self._fds[str(path)]
        n = struct.unpack("<Q", os.pread(fd, 8, 0))[0]
        header_bytes = os.pread(fd, n, 8)
        header = json.loads(header_bytes)
        if tensor_name not in header:
            return
        info = header[tensor_name]
        dtype_str = info["dtype"]
        shape = info["shape"]
        offsets = info["data_offsets"]
        start, end = offsets
        total_bytes = end - start
        if total_bytes == 0:
            return
        np_dtype = self.DTYPE_MAP.get(dtype_str)
        if np_dtype is None:
            return
        # Determine number of experts (E)
        if len(shape) < 1:
            return
        num_experts = shape[0]
        per_expert_nbytes = total_bytes // num_experts
        # Determine parts (cols_part)
        if len(shape) == 3:
            cols_part = shape[2]
        elif len(shape) == 2:
            cols_part = shape[1]
        else:
            return
        
        # Map (layer, proj, part) -> metadata
        # We need to handle parts. The problem says "part" in the mapping key.
        # For gate/up/down, there is typically one part unless split.
        # Let's assume standard case: one part per tensor definition.
        # However, the prompt says "(layer, proj, part) -> ...".
        # If the tensor is split across multiple files or parts, we might need to handle that.
        # But the index maps tensor name to file.
        # Let's assume one part per tensor for now, or derive parts from shape.
        # Actually, looking at the read method: "submit one task per (part, expert)".
        # This implies a tensor might be split into 'parts'.
        # In standard MoE, gate_proj, up_proj, down_proj are single tensors per layer.
        # But sometimes they are sharded.
        # Let's assume for this implementation that each tensor in the index is one 'part'.
        # If a tensor has shape [E, R, C], it's one part.
        # If it has shape [E, R, C1, C2], it might be two parts?
        # The prompt says "shape [E, rows, cols_part]". This suggests cols_part is a single dimension.
        # So we treat each tensor entry as one part.
        
        key = (layer, proj, 0) # Assuming part 0 for single tensor
        self._tensors[key] = {
            "file_path": str(path),
            "abs_offset": 8 + n + start,
            "per_expert_nbytes": per_expert_nbytes,
            "per_expert_shape": shape[1:],
            "np_dtype": np_dtype,
        }

    def read(self, layer: int, proj: str, experts: np.ndarray) -> Dict[str, np.ndarray]:
        results: Dict[str, np.ndarray] = {"weight": None, "scales": None, "biases": None}
        futures: List[Any] = []
        
        # Identify which tensors we need
        needed_parts = []
        for p in ("gate_proj", "up_proj", "down_proj"):
            if p == proj:
                key = (layer, p, 0)
                if key not in self._tensors:
                    raise KeyError(f"Tensor (layer={layer}, proj={proj}) not found in index.")
                needed_parts.append((p, self._tensors[key]))
        
        if not needed_parts:
            raise KeyError(f"No switch_mlp tensors found for layer={layer}, proj={proj}")

        # Allocate output arrays
        num_experts = len(experts)
        out_weight = np.empty((num_experts,) + needed_parts[0][1]["per_expert_shape"], dtype=needed_parts[0][1]["np_dtype"])
        out_scales = np.empty((num_experts,) + needed_parts[1][1]["per_expert_shape"], dtype=needed_parts[1][1]["np_dtype"])
        out_biases = np.empty((num_experts,) + needed_parts[2][1]["per_expert_shape"], dtype=needed_parts[2][1]["np_dtype"])
        
        results["weight"] = out_weight
        results["scales"] = out_scales
        results["biases"] = out_biases

        # Submit tasks
        for part_name, meta in needed_parts:
            fd = self._fds[meta["file_path"]]
            base_offset = meta["abs_offset"]
            per_expert_nbytes = meta["per_expert_nbytes"]
            out_arr = results[part_name]
            
            for i, e_idx in enumerate(experts):
                offset = base_offset + e_idx * per_expert_nbytes
                # Create a memoryview of the specific expert slice
                # out_arr[i] has shape per_expert_shape
                # We need to cast it to bytes
                mv = memoryview(out_arr[i]).cast("B")
                futures.append(self._pool.submit(self._read_expert, fd, mv, offset, per_expert_nbytes))

        # Wait for all and check errors
        for f in futures:
            f.result() # Raises OSError if failed

        return results

    @staticmethod
    def _read_expert(fd: int, mv: memoryview, offset: int, expected_nbytes: int) -> None:
        data = os.pread(fd, expected_nbytes, offset)
        if len(data) != expected_nbytes:
            raise OSError(f"Read {len(data)} bytes, expected {expected_nbytes} at offset {offset}")
        mv[:expected_nbytes] = data

    def close(self) -> None:
        self._pool.shutdown(wait=True)
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()
```