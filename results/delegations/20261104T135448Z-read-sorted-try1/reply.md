```python
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
```