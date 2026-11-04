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
                "num_experts": shape[0],
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

    def num_experts(self, layer: int, proj: str) -> int:
        """Number of experts E of (layer, proj) (from the "weight" entry). KeyError if unknown."""
        return self._tensor_info[(layer, proj, "weight")]["num_experts"]

    def read_sorted(self, layer: int, proj: str, experts: np.ndarray) -> dict:
        """Same result as read(): {"weight": arr, "scales": arr, "biases": arr}, each [len(experts), *per_expert_shape],
        experts in the given order. REQUIREMENT: experts is strictly increasing (raise ValueError otherwise).
        Reads in DISK ORDER with few large requests:
          1. Split experts into runs of consecutive ids, e.g. [5,6,7,40,41,200] -> runs (slot 0, ids 5..7),
             (slot 3, ids 40..41), (slot 5, id 200). A run of k experts is ONE contiguous byte range on disk
             (k * per_expert_nbytes bytes at abs_offset + first_id * per_expert_nbytes) and ONE contiguous
             destination out[slot:slot+k].
          2. Cut every run into chunks of at most CHUNK = 32 * 1024 * 1024 bytes.
          3. Build the list of all chunks of all three parts, SORT it by (file_name, file offset), submit each to
             self._executor: os.preadv(fd, [dest_view[a:b]], file_offset) where
             dest_view = memoryview(out[slot:slot+k]).cast("B"); check the returned byte count == b - a
             (raise OSError otherwise).
          4. Wait for every future (re-raise errors), return the dict."""
        CHUNK = 32 * 1024 * 1024
        if len(experts) == 0:
            return {"weight": np.empty((0, *self._tensor_info[(layer, proj, "weight")]["per_expert_shape"]), dtype=self._tensor_info[(layer, proj, "weight")]["np_dtype"]),
                    "scales": np.empty((0, *self._tensor_info[(layer, proj, "scales")]["per_expert_shape"]), dtype=self._tensor_info[(layer, proj, "scales")]["np_dtype"]),
                    "biases": np.empty((0, *self._tensor_info[(layer, proj, "biases")]["per_expert_shape"]), dtype=self._tensor_info[(layer, proj, "biases")]["np_dtype"])}
        if not np.all(np.diff(experts) > 0):
            raise ValueError("experts must be strictly increasing")
        futures = []
        for part in ("weight", "scales", "biases"):
            info = self._tensor_info[(layer, proj, part)]
            out = np.empty((len(experts), *info["per_expert_shape"]), dtype=info["np_dtype"])
            runs = self._split_into_runs(experts)
            for slot, run_experts in runs:
                k = len(run_experts)
                per_expert_nbytes = info["per_expert_nbytes"]
                abs_offset = info["abs_offset"]
                file_name = info["file_name"]
                fd = self._fds[file_name]
                start_byte = abs_offset + run_experts[0] * per_expert_nbytes
                total_bytes = k * per_expert_nbytes
                dest_view = memoryview(out[slot:slot + k]).cast("B")
                off = 0
                while off < total_bytes:
                    chunk_size = min(CHUNK, total_bytes - off)
                    a = off
                    b = off + chunk_size
                    futures.append(self._executor.submit(self._read_chunk, fd, dest_view, a, b, start_byte + off))
                    off = b
            if part == "weight":
                weight_arr = out
            elif part == "scales":
                scales_arr = out
            else:
                biases_arr = out
        for f in futures:
            f.result()
        return {"weight": weight_arr, "scales": scales_arr, "biases": biases_arr}

    def _split_into_runs(self, experts: np.ndarray) -> list:
        """Split strictly increasing experts into runs of consecutive ids."""
        if len(experts) == 0:
            return []
        runs = []
        start = 0
        for i in range(1, len(experts)):
            if experts[i] != experts[i - 1] + 1:
                runs.append((start, experts[start:i].tolist()))
                start = i
        runs.append((start, experts[start:].tolist()))
        return runs

    def _read_chunk(self, fd, dest_view, a, b, file_offset):
        """Read a chunk from disk into dest_view[a:b]."""
        n = os.preadv(fd, [dest_view[a:b]], file_offset)
        if n != b - a:
            raise OSError(f"Expected to read {b - a} bytes, got {n}")
        return n

    def close(self) -> None:
        self._executor.shutdown(wait=True)
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()
