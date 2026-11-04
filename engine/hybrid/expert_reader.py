import json
import os
import re
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np

NAME = re.compile(
    r"layers\.(\d+)\.mlp\.switch_mlp\.(gate_proj|up_proj|down_proj)\.(weight|scales|biases)$"
)


class ExpertReader:
    """Read selected MoE experts from MLX safetensors shards in parallel."""

    def __init__(self, model_dir: Path, threads: int = 8) -> None:
        self._model_dir = model_dir
        self._executor = ThreadPoolExecutor(max_workers=threads)
        self._fds: Dict[str, int] = {}
        self._tensor_info: Dict[Tuple[int, str, str], Dict] = {}
        self._file_headers: Dict[str, Dict] = {}
        self.n_layers: int = 0

        self._build_index()

    def _build_index(self) -> None:
        index_path = self._model_dir / "model.safetensors.index.json"
        with open(index_path, "r") as f:
            index = json.load(f)

        weight_map = index.get("weight_map", {})
        layers: set[int] = set()

        for tensor_name, file_name in weight_map.items():
            m = NAME.search(tensor_name)
            if not m:
                continue

            layer = int(m.group(1))
            proj = m.group(2)
            part = m.group(3)
            layers.add(layer)

            shard_path = self._model_dir / file_name
            if not shard_path.exists():
                raise FileNotFoundError(f"Shard not found: {shard_path}")

            if file_name not in self._fds:
                self._fds[file_name] = os.open(str(shard_path), os.O_RDONLY)

            header, header_len = self._get_header(shard_path)
            tensor_meta = header.get(tensor_name)
            if tensor_meta is None:
                raise KeyError(f"{tensor_name} listed in the index but missing from {file_name}")

            dtype_str = tensor_meta["dtype"]
            shape = tuple(tensor_meta["shape"])
            start, end = tensor_meta["data_offsets"]

            np_dtype = np.uint32 if dtype_str == "U32" else np.uint16
            per_expert_shape = tuple(shape[1:])
            per_expert_nbytes = (end - start) // shape[0]
            abs_offset = 8 + header_len + start

            self._tensor_info[(layer, proj, part)] = {
                "file_name": file_name,
                "abs_offset": abs_offset,
                "per_expert_nbytes": per_expert_nbytes,
                "per_expert_shape": per_expert_shape,
                "np_dtype": np_dtype,
            }

        self.n_layers = len(layers)

    def _get_header(self, shard_path: Path) -> tuple[dict, int]:
        if str(shard_path) in self._file_headers:
            return self._file_headers[str(shard_path)]

        fd = os.open(str(shard_path), os.O_RDONLY)
        try:
            n_bytes = os.pread(fd, 8, 0)
            if len(n_bytes) != 8:
                raise OSError("Unexpected file size")
            header_len = struct.unpack("<Q", n_bytes)[0]
            header_bytes = os.pread(fd, header_len, 8)
            header = (json.loads(header_bytes), header_len)
        finally:
            os.close(fd)

        self._file_headers[str(shard_path)] = header
        return header

    def read(self, layer: int, proj: str, experts: np.ndarray) -> dict:
        if (layer, proj, "weight") not in self._tensor_info:
            raise KeyError(f"Tensor (layer={layer}, proj={proj}) not found in index.")

        futures = []
        results: Dict[str, np.ndarray] = {}

        for part in ("weight", "scales", "biases"):
            info = self._tensor_info.get((layer, proj, part))
            if info is None:
                raise KeyError(f"Missing part '{part}' for (layer={layer}, proj={proj})")

            out = np.empty((len(experts),) + info["per_expert_shape"], dtype=info["np_dtype"])
            results[part] = out

            for i, e in enumerate(experts):
                offset = info["abs_offset"] + int(e) * info["per_expert_nbytes"]
                fd = self._fds[info["file_name"]]
                futures.append(
                    self._executor.submit(
                        self._read_expert,
                        fd,
                        out,
                        i,
                        offset,
                        info["per_expert_nbytes"],
                        part,
                    )
                )

        for fut in futures:
            fut.result()

        return results

    @staticmethod
    def _read_expert(
        fd: int,
        out: np.ndarray,
        i: int,
        offset: int,
        nbytes: int,
        part: str,
    ) -> None:
        buf = memoryview(out[i]).cast("B")
        got = os.preadv(fd, [buf], offset)   # returns the byte count
        if got != nbytes:
            raise OSError(f"read {got} bytes instead of {nbytes} for {part} expert slot {i}")

    def close(self) -> None:
        self._executor.shutdown(wait=True)
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()
