"""Benchmark vbeam PICMUS plane-wave DAS on CPU/JAX.

The goal is to separate data/setup time, beamformer construction time, JIT compile
plus first execution time, and repeated execution time. The default cases are chosen
for teaching on a CPU machine; use --include-native-full only when you are ready for
a much heavier run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import csv
import os
import site
import sys
import time
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".matplotlib"))
os.environ.setdefault("PYTHONNOUSERSITE", "1")

USER_SITE = Path(site.getusersitepackages()).resolve()
sys.path = [
    path
    for path in sys.path
    if not path or Path(path).resolve() != USER_SITE
]

import jax
from pyuff_ustb import Uff

from vbeam.beamformers import get_das_beamformer
from vbeam.data_importers import import_pyuff
from vbeam.fastmath import backend_manager
from vbeam.util.download import cached_download


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


def default_cases(include_native_full: bool) -> list[BenchmarkCase]:
    cases = [
        BenchmarkCase("resize_48x64_tx1", 48, 64, 1),
        BenchmarkCase("resize_48x64_tx5", 48, 64, 5),
        BenchmarkCase("resize_96x128_tx15", 96, 128, 15),
        BenchmarkCase("resize_96x128_tx75", 96, 128, 75),
        BenchmarkCase("native_scan_tx1", None, None, 1),
    ]
    if include_native_full:
        cases.append(BenchmarkCase("native_scan_tx75", None, None, 75))
    return cases


def seconds_since(start: float) -> float:
    return time.perf_counter() - start


def time_call(fn) -> tuple[float, object]:
    start = time.perf_counter()
    value = fn()
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
    return seconds_since(start), value


def prepare_setup(channel_data, scan, case: BenchmarkCase):
    setup = import_pyuff(channel_data, scan, frames=0)
    if case.scan_x is not None and case.scan_z is not None:
        setup.scan = setup.scan.resize(x=case.scan_x, z=case.scan_z)
    if case.n_tx != setup.signal.shape[0]:
        setup.signal = setup.signal[: case.n_tx]
        setup.wave_data = setup.wave_data[: case.n_tx]
    return setup


def run_case(channel_data, scan, case: BenchmarkCase) -> dict:
    if hasattr(jax, "clear_caches"):
        jax.clear_caches()

    setup_time, setup = time_call(lambda: prepare_setup(channel_data, scan, case))
    sizes = setup.size()

    build_start = time.perf_counter()
    beamformer = jax.jit(
        get_das_beamformer(
            setup,
            compensate_for_apodization_overlap=True,
            log_compress=True,
            scan_convert=True,
        )
    )
    build_time = seconds_since(build_start)

    first_run, result = time_call(lambda: beamformer(**setup.data))
    second_run, _ = time_call(lambda: beamformer(**setup.data))
    third_run, _ = time_call(lambda: beamformer(**setup.data))

    device = jax.devices()[0]

    return {
        "backend": jax.default_backend(),
        "jax_version": jax.__version__,
        "device": getattr(device, "device_kind", str(device)),
        "case": case.name,
        "scan": case.scan_label,
        "points": sizes["points"],
        "receivers": sizes["receivers"],
        "transmits": sizes["transmits"],
        "signal_time": sizes["signal_time"],
        "setup_s": setup_time,
        "build_s": build_time,
        "first_run_s": first_run,
        "second_run_s": second_run,
        "third_run_s": third_run,
        "steady_mean_s": (second_run + third_run) / 2,
        "result_shape": "x".join(str(x) for x in result.shape),
        "result_dtype": str(result.dtype),
    }


def print_table(rows: list[dict]) -> None:
    headers = [
        "case",
        "scan",
        "transmits",
        "points",
        "setup_s",
        "build_s",
        "first_run_s",
        "second_run_s",
        "third_run_s",
        "steady_mean_s",
        "result_shape",
    ]
    widths = {
        header: max(
            len(header),
            *[
                len(f"{row[header]:.3f}")
                if isinstance(row[header], float)
                else len(str(row[header]))
                for row in rows
            ],
        )
        for header in headers
    }
    print()
    print("Benchmark results")
    print(" ".join(header.ljust(widths[header]) for header in headers))
    print(" ".join("-" * widths[header] for header in headers))
    for row in rows:
        values = []
        for header in headers:
            value = row[header]
            if isinstance(value, float):
                value = f"{value:.3f}"
            values.append(str(value).ljust(widths[header]))
        print(" ".join(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--include-native-full",
        action="store_true",
        help="Also run native scan with all 75 transmits. This can be very slow on CPU.",
    )
    args = parser.parse_args()

    backend_manager.active_backend = "jax"
    output_dir = SCRIPT_DIR / "outputs"
    output_dir.mkdir(exist_ok=True)

    print(f"Python executable: {sys.executable}")
    backend = jax.default_backend()
    print(f"JAX version: {jax.__version__}")
    print(f"JAX backend: {backend}")
    print(f"JAX devices: {jax.devices()}")
    print("Downloading/loading dataset...")
    download_time, uff_path = time_call(
        lambda: cached_download(DATA_URL, local_dir=str(SCRIPT_DIR / "data"))
    )
    print(f"UFF path: {uff_path}")
    print(f"download/cache time: {download_time:.3f} s")

    print("Reading UFF channel_data and scan...")
    read_start = time.perf_counter()
    uff = Uff(uff_path)
    channel_data = uff.read("/channel_data")
    scan = uff.read("/scan")
    read_time = seconds_since(read_start)
    print(f"UFF read time: {read_time:.3f} s")
    print(
        "raw channel_data.data shape:",
        getattr(channel_data.data, "shape", None),
        "scan:",
        type(scan).__name__,
    )

    rows = []
    for case in default_cases(args.include_native_full):
        print()
        print(f"Running case: {case.name} scan={case.scan_label} n_tx={case.n_tx}")
        rows.append(run_case(channel_data, scan, case))

    print_table(rows)

    output_path = output_dir / f"picmus_jax_benchmark_{backend}.csv"
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print()
    print(f"Saved CSV: {output_path}")


if __name__ == "__main__":
    main()
