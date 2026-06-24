"""Shared utilities for the vbeam learning benchmark scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import os
import platform
import socket
import subprocess
import sys
import time
from typing import Callable, Iterable, Optional, Sequence

DATA_URL = "http://www.ustb.no/datasets/PICMUS_carotid_cross.uff"


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    scan_x: Optional[int]
    scan_z: Optional[int]
    n_tx: int

    @property
    def scan_label(self) -> str:
        if self.scan_x is None or self.scan_z is None:
            return "native"
        return f"{self.scan_x}x{self.scan_z}"


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise ValueError("value must be >= 1")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise ValueError("value must be >= 0")
    return parsed


def seconds_since(start: float) -> float:
    return time.perf_counter() - start


def time_call(fn: Callable[[], object]) -> tuple[float, object]:
    start = time.perf_counter()
    value = fn()
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
    return seconds_since(start), value


def select_cases(cases: Sequence[BenchmarkCase], selected: Optional[str]) -> list[BenchmarkCase]:
    if not selected or selected.lower() == "all":
        return list(cases)

    requested = [name.strip() for name in selected.split(",") if name.strip()]
    by_name = {case.name: case for case in cases}
    unknown = [name for name in requested if name not in by_name]
    if unknown:
        valid = ", ".join(by_name)
        raise ValueError(
            f"Unknown case(s): {', '.join(unknown)}. Valid cases: {valid}"
        )
    return [by_name[name] for name in requested]


def select_names(
    available: Sequence[tuple[str, object]], selected: Optional[str], label: str
) -> list[tuple[str, object]]:
    if not selected or selected.lower() == "all":
        return list(available)

    requested = [name.strip() for name in selected.split(",") if name.strip()]
    by_name = {name: value for name, value in available}
    unknown = [name for name in requested if name not in by_name]
    if unknown:
        valid = ", ".join(by_name)
        raise ValueError(
            f"Unknown {label}(s): {', '.join(unknown)}. Valid {label}s: {valid}"
        )
    return [(name, by_name[name]) for name in requested]


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return math.nan
    if len(values) == 1:
        return float(values[0])

    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * p
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def compute_timing_stats(times: Sequence[float]) -> dict[str, float | int]:
    if not times:
        raise ValueError("At least one timed repeat is required")

    ordered = sorted(float(value) for value in times)
    return {
        "steady_repeats": len(times),
        "steady_mean_s": sum(times) / len(times),
        "steady_median_s": percentile(ordered, 0.5),
        "steady_min_s": ordered[0],
        "steady_max_s": ordered[-1],
        "steady_p90_s": percentile(ordered, 0.9),
    }


def resolve_uff_path(
    *,
    data_path: Optional[Path],
    data_dir: Path,
    url: str = DATA_URL,
) -> tuple[Path, float]:
    if data_path is not None:
        path = data_path.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return path, 0.0

    from vbeam.util.download import cached_download

    elapsed, path = time_call(lambda: cached_download(url, local_dir=str(data_dir)))
    return Path(path), elapsed


def first_or_empty(values: Sequence[float], index: int) -> float | str:
    if index >= len(values):
        return ""
    return values[index]


def run_warmups(fn: Callable[[], object], count: int) -> None:
    for _ in range(count):
        time_call(fn)


def timed_repeats(fn: Callable[[], object], count: int) -> tuple[list[float], object]:
    times: list[float] = []
    last_result = None
    for _ in range(count):
        elapsed, last_result = time_call(fn)
        times.append(elapsed)
    return times, last_result


def benchmark_metadata(jax_module, data_path: Path) -> dict[str, str]:
    devices = jax_module.devices()
    device = devices[0] if devices else ""
    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "backend": jax_module.default_backend(),
        "jax_version": jax_module.__version__,
        "jaxlib_version": getattr(jax_module.lib, "__version__", ""),
        "device": getattr(device, "device_kind", str(device)),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "xla_python_client_preallocate": os.environ.get(
            "XLA_PYTHON_CLIENT_PREALLOCATE", ""
        ),
        "data_path": str(data_path),
    }


def nvidia_smi_snapshot(enabled: bool) -> str:
    if not enabled:
        return ""
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return " | ".join(line.strip() for line in completed.stdout.splitlines())


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    import csv

    rows = list(rows)
    if not rows:
        raise ValueError("No benchmark rows to write")

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    path.parent.mkdir(exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
