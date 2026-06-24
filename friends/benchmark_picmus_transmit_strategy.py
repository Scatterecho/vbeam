"""Compare transmit reduction strategies for vbeam PICMUS DAS on CPU/JAX.

This script compares the default vbeam DAS strategy, which uses Reduce.Sum over
transmits, against a vmap-all strategy that exposes the transmits dimension before
applying a named-axis sum. The latter is conceptually more materialized, although XLA
may still fuse parts of the computation.
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
from typing import Callable, Optional

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
import psutil
from pyuff_ustb import Uff

from vbeam.beamformers import get_das_beamformer
from vbeam.beamformers.building_blocks import (
    sum_over_dimensions,
    unflatten_points,
    vectorize_over_datacube,
)
from vbeam.core import signal_for_point
from vbeam.data_importers import import_pyuff
from vbeam.fastmath import backend_manager
from vbeam.util.download import cached_download
from vbeam.util.transformations import compose


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


def rss_gb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024**3


def seconds_since(start: float) -> float:
    return time.perf_counter() - start


def time_call(fn):
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


def get_vmap_all_then_sum_beamformer(setup):
    return compose(
        signal_for_point,
        vectorize_over_datacube(setup),
        sum_over_dimensions(setup),
        unflatten_points(setup),
    ).build(setup.spec)


def default_reduce_builder(setup):
    return get_das_beamformer(
        setup,
        compensate_for_apodization_overlap=False,
        log_compress=False,
        scan_convert=False,
    )


def vmap_all_builder(setup):
    return get_vmap_all_then_sum_beamformer(setup)


def run_strategy(
    channel_data,
    scan,
    case: BenchmarkCase,
    strategy_name: str,
    builder: Callable,
) -> dict:
    if hasattr(jax, "clear_caches"):
        jax.clear_caches()

    setup_time, setup = time_call(lambda: prepare_setup(channel_data, scan, case))
    sizes = setup.size()
    conceptual_cube_gb = (
        sizes["points"] * sizes["receivers"] * sizes["transmits"] * 8 / 1024**3
    )

    rss_before_build = rss_gb()
    build_start = time.perf_counter()
    beamformer = jax.jit(builder(setup))
    build_time = seconds_since(build_start)
    rss_after_build = rss_gb()

    first_run, result = time_call(lambda: beamformer(**setup.data))
    rss_after_first = rss_gb()
    second_run, result2 = time_call(lambda: beamformer(**setup.data))
    rss_after_second = rss_gb()
    third_run, _ = time_call(lambda: beamformer(**setup.data))
    rss_after_third = rss_gb()

    result.block_until_ready()
    result2.block_until_ready()
    device = jax.devices()[0]

    return {
        "backend": jax.default_backend(),
        "jax_version": jax.__version__,
        "device": getattr(device, "device_kind", str(device)),
        "case": case.name,
        "strategy": strategy_name,
        "scan": case.scan_label,
        "points": sizes["points"],
        "receivers": sizes["receivers"],
        "transmits": sizes["transmits"],
        "conceptual_cube_gb": conceptual_cube_gb,
        "setup_s": setup_time,
        "build_s": build_time,
        "first_run_s": first_run,
        "second_run_s": second_run,
        "third_run_s": third_run,
        "steady_mean_s": (second_run + third_run) / 2,
        "rss_before_build_gb": rss_before_build,
        "rss_after_build_gb": rss_after_build,
        "rss_after_first_gb": rss_after_first,
        "rss_after_second_gb": rss_after_second,
        "rss_after_third_gb": rss_after_third,
        "result_shape": "x".join(str(x) for x in result.shape),
        "result_dtype": str(result.dtype),
    }


def print_table(rows: list[dict]) -> None:
    headers = [
        "case",
        "strategy",
        "scan",
        "transmits",
        "points",
        "conceptual_cube_gb",
        "first_run_s",
        "second_run_s",
        "third_run_s",
        "steady_mean_s",
        "rss_after_third_gb",
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
    print("Transmit strategy benchmark")
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--native",
        action="store_true",
        help="Also benchmark native scan with 75 transmits.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backend_manager.active_backend = "jax"
    output_dir = SCRIPT_DIR / "outputs"
    output_dir.mkdir(exist_ok=True)

    print(f"Python executable: {sys.executable}")
    backend = jax.default_backend()
    print(f"JAX version: {jax.__version__}")
    print(f"JAX backend: {backend}")
    print(f"JAX devices: {jax.devices()}")
    print("Downloading/loading dataset...")
    _, uff_path = time_call(
        lambda: cached_download(DATA_URL, local_dir=str(SCRIPT_DIR / "data"))
    )
    print(f"UFF path: {uff_path}")

    print("Reading UFF channel_data and scan...")
    uff = Uff(uff_path)
    channel_data = uff.read("/channel_data")
    scan = uff.read("/scan")
    print("raw channel_data.data shape:", getattr(channel_data.data, "shape", None))

    cases = [BenchmarkCase("resize_96x128_tx75", 96, 128, 75)]
    if args.native:
        cases.append(BenchmarkCase("native_scan_tx75", None, None, 75))

    strategies = [
        ("default_reduce_sum", default_reduce_builder),
        ("vmap_all_then_sum", vmap_all_builder),
    ]

    rows = []
    for case in cases:
        for strategy_name, builder in strategies:
            print()
            print(
                f"Running case={case.name} strategy={strategy_name} "
                f"scan={case.scan_label} n_tx={case.n_tx}"
            )
            rows.append(run_strategy(channel_data, scan, case, strategy_name, builder))

    print_table(rows)

    output_path = output_dir / f"picmus_transmit_strategy_benchmark_{backend}.csv"
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print()
    print(f"Saved CSV: {output_path}")


if __name__ == "__main__":
    main()
