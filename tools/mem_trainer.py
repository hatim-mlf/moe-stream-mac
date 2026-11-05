"""Run a one-iteration measurement through mlx_lm.tuner.trainer.train.

TrainingArgs, CacheDataset, and optimizer setup mirror mlx_lm/lora.py:271-313;
the train/eval callback order and step-1 evaluation mirror
mlx_lm/tuner/trainer.py:273-360. The loss and compile scope mirror
tools/ft_lora_lowmem.py:19-33.
"""

from __future__ import annotations

import importlib
import threading
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

if __package__:
    profile = importlib.import_module(f"{__package__}.mem_profile")
else:
    profile = importlib.import_module("mem_profile")

WIRED_SAMPLE_SECONDS = 0.2


class SystemWiredSampler:
    def __init__(self, read_wired: Callable[[], int]) -> None:
        self.read_wired = read_wired
        self.maximum = 0
        self.error: Exception | None = None
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def _sample(self) -> None:
        while not self.stop_event.wait(WIRED_SAMPLE_SECONDS):
            try:
                value = self.read_wired()
            except Exception as error:
                self.error = error
                self.stop_event.set()
                return
            self.maximum = max(self.maximum, value)

    def start(self) -> None:
        if self.thread is not None:
            raise RuntimeError("wired sampler already started")
        self.maximum = self.read_wired()
        self.thread = threading.Thread(target=self._sample, name="system-wired-sampler", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()
        if self.error is not None:
            raise RuntimeError("system wired-memory sampler failed") from self.error


class _SingleRowDataset:
    def __init__(self, tokens: list[int], offset: int) -> None:
        self.row = (tokens, offset)

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> tuple[list[int], int]:
        if index != 0:
            raise IndexError(index)
        return self.row

    def process(self, item: tuple[list[int], int]) -> tuple[list[int], int]:
        return item


def _training_args(config: Mapping[str, Any], adapter_file: Path, checkpoint: bool) -> Any:
    from mlx_lm.tuner.trainer import TrainingArgs

    return TrainingArgs(
        batch_size=int(config["batch_size"]), iters=1,
        val_batches=int(config["val_batches"]),
        steps_per_report=int(config["steps_per_report"]),
        steps_per_eval=int(config["steps_per_eval"]),
        steps_per_save=int(config["save_every"]),
        adapter_file=str(adapter_file),
        max_seq_length=int(config["max_seq_length"]),
        grad_checkpoint=checkpoint,
        grad_accumulation_steps=int(config["grad_accumulation_steps"]),
        clear_cache_threshold=int(config["clear_cache_threshold"]),
    )


def _training_config(config: Mapping[str, Any], checkpoint: bool) -> dict[str, Any]:
    keys = (
        "batch_size", "val_batches", "steps_per_report", "steps_per_eval",
        "save_every", "max_seq_length", "grad_accumulation_steps",
        "clear_cache_threshold", "optimizer",
    )
    details = {key: config[key] for key in keys}
    details.update({"iters": 1, "grad_checkpoint": checkpoint})
    return details


def _optimizer(optim: Any, lora: Any, config: Mapping[str, Any]) -> Any:
    lr = lora.build_schedule(config["lr_schedule"]) if config["lr_schedule"] else config["learning_rate"]
    optimizer_name = str(config["optimizer"]).lower()
    optimizer_classes = {
        "adam": optim.Adam, "adamw": optim.AdamW, "muon": optim.Muon,
        "sgd": optim.SGD, "adafactor": optim.Adafactor,
    }
    if optimizer_name not in optimizer_classes:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    optimizer_config = config["optimizer_config"].get(optimizer_name, {})
    return optimizer_classes[optimizer_name](learning_rate=lr, **optimizer_config)


def _callback(
    mx: Any,
    config: Mapping[str, Any],
    started: float,
    sampler: SystemWiredSampler,
    on_measurement: Callable[[dict[str, Any]], None] | None,
) -> Any:
    base = importlib.import_module("mlx_lm.tuner.callbacks").TrainingCallback

    class MemoryCallback(base):
        def __init__(self) -> None:
            self.measurements: list[dict[str, Any]] = []

        def _record(self, phase: str, details: Mapping[str, Any]) -> None:
            record = profile._emit(
                config, profile._memory_measurement(mx, phase, started), on_measurement,
                trainer_details=dict(details),
                max_system_wired_bytes=sampler.maximum,
            )
            self.measurements.append(record)

        def on_val_loss_report(self, val_info: dict[str, Any]) -> None:
            self._record("after_eval", val_info)

        def on_train_loss_report(self, train_info: dict[str, Any]) -> None:
            self._record("after_train_step", train_info)

    return MemoryCallback()


def _trainer_modules() -> tuple[Any, Any, tuple[Any, Any]]:
    mx, _, optim = profile._import_mlx()
    from mlx_lm import lora
    from mlx_lm.tuner.trainer import train

    return mx, optim, (lora, train)


def _trainer_run_config(
    tokens: int,
    layers: int,
    gdn_checkpoint: int,
    checkpoint: bool,
    prepared: bool,
    yaml_config: Mapping[str, Any],
    row_source: str | Path | Iterable[Mapping[str, Any]] | None,
    cache_limit_gb: float | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from mlx_lm import lora

    config = {**lora.CONFIG_DEFAULTS, **yaml_config}
    run_config = profile._make_config(
        "trainer-step", tokens, layers, gdn_checkpoint, checkpoint, prepared,
        batch_size=int(config["batch_size"]), iters=1,
        val_batches=int(config["val_batches"]),
        learning_rate=float(config["learning_rate"]),
        lora_parameters=config["lora_parameters"],
        training_args=_training_config(config, checkpoint),
        row_source=profile._row_source_name(row_source), completion_mask="last_half",
        trainer="mlx_lm.tuner.trainer.train",
        cache_limit_gb=profile._cache_limit_config(cache_limit_gb),
    )
    return config, run_config


def run_trainer_step(
    tokens: int,
    layers: int,
    *,
    gdn_checkpoint: int = 1,
    grad_checkpoint: bool | None = None,
    model: Any | None = None,
    tokenizer: Any | None = None,
    row_source: str | Path | Iterable[Mapping[str, Any]] | None = None,
    model_prepared: bool = True,
    cache_limit_gb: float | None = None,
    on_measurement: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if tokens < 2 or layers < 1 or gdn_checkpoint not in (0, 1):
        raise ValueError("tokens must be >= 2, layers >= 1, and gdn_checkpoint must be 0 or 1")
    yaml_config = profile.load_training_config()
    checkpoint = bool(yaml_config["grad_checkpoint"] if grad_checkpoint is None else grad_checkpoint)
    prepared = model is not None and model_prepared
    setup = lambda selected, values: profile._prepare_training(
        selected, layers, values, prepared, False
    )
    mem_guard = profile._import_local_module("mem_guard")
    sampler = SystemWiredSampler(mem_guard.read_wired_bytes)
    try:
        mx, model, tokenizer, load_metrics, _ = profile._load_model(
            gdn_checkpoint, model, tokenizer, setup=setup, training_config=yaml_config,
            cache_limit_gb=cache_limit_gb, on_runtime_ready=sampler.start,
        )
        config, run_config = _trainer_run_config(
            tokens, layers, gdn_checkpoint, checkpoint, prepared, yaml_config,
            row_source, cache_limit_gb,
        )
        load_record = profile._emit(
            run_config, load_metrics, on_measurement,
            max_system_wired_bytes=sampler.maximum,
        )
        row, _ = profile.build_row(tokens, tokenizer, row_source)
        train_data, valid_data = _trainer_datasets(row, tokens // 2)
        records = [load_record]
        return _run_trainer(
            mx, model, config, checkpoint, run_config, train_data, valid_data,
            records, sampler, on_measurement,
        )
    finally:
        sampler.stop()


def _trainer_datasets(row: list[int], offset: int) -> tuple[Any, Any]:
    cache_dataset = importlib.import_module("mlx_lm.tuner.datasets").CacheDataset
    source = _SingleRowDataset(row, offset)
    return cache_dataset(source), cache_dataset(source)


def _run_trainer(
    mx: Any,
    model: Any,
    config: Mapping[str, Any],
    checkpoint: bool,
    run_config: Mapping[str, Any],
    train_data: Any,
    valid_data: Any,
    records: list[dict[str, Any]],
    sampler: SystemWiredSampler,
    on_measurement: Callable[[dict[str, Any]], None] | None,
) -> dict[str, Any]:
    mx, optim, modules = _trainer_modules()
    lora, trainer = modules
    with tempfile.TemporaryDirectory(prefix="mem-trainer-step-") as temporary:
        args = _training_args(config, Path(temporary) / "adapters.safetensors", checkpoint)
        optimizer = _optimizer(optim, lora, config)
        started = time.perf_counter()
        memory_callback = _callback(mx, run_config, started, sampler, on_measurement)
        mx.disable_compile()
        try:
            trainer(
                model=model, optimizer=optimizer, train_dataset=train_data,
                val_dataset=valid_data, args=args,
                loss=profile._import_local_module("ft_lowmem_loss").completion_only_loss,
                training_callback=memory_callback,
            )
        finally:
            mx.enable_compile()
        mx.eval(model.parameters(), optimizer.state)
        records.extend(memory_callback.measurements)
        events = {record["phase"] for record in memory_callback.measurements}
        if events != {"after_eval", "after_train_step"}:
            raise RuntimeError("trainer did not report both eval and train-step callbacks")
        sampler.stop()
        overall = profile._memory_measurement(mx, "overall", started)
        records.append(profile._emit(
            run_config, overall, on_measurement,
            max_system_wired_bytes=sampler.maximum,
        ))
    return {"config": dict(run_config), "phases": records}
