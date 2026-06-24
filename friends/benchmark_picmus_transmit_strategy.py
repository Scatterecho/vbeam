"""Compare transmit reduction strategies for vbeam PICMUS DAS on JAX."""

from __future__ import annotations

from pathlib import Path
import argparse
import os
import site
import sys
import time
from typing import Callable

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

from benchmark_helpers import (
    BenchmarkCase,
    benchmark_metadata,
    compute_timing_stats,
    first_or_empty,
    nonnegative_int,
    nvidia_smi_snapshot,
    positive_int,
    resolve_uff_path,
    run_warmups,
    seconds_since,
    select_cases,
    select_names,
    time_call,
    timed_repeats,
    write_csv,
)
from vbeam.beamformers import get_das_beamformer
from vbeam.beamformers.building_blocks import (
    sum_over_dimensions,
    unflatten_points,
    vectorize_over_datacube,
)
from vbeam.core import signal_for_point
from vbeam.data_importers import import_pyuff
from vbeam.fastmath import backend_manager
from vbeam.util.transformations import compose


ALL_CASES = [
    BenchmarkCase("resize_96x128_tx75", 96, 128, 75),
    BenchmarkCase("native_scan_tx75", None, None, 75),
]


def rss_gb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024**3


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


STRATEGIES = [
    ("default_reduce_sum", default_reduce_builder),
    ("vmap_all_then_sum", vmap_all_builder),
]


def run_strategy(
    channel_data,
    scan,
    case: BenchmarkCase,
    strategy_name: str,
    builder: Callable,
    *,
    warmups: int,
    repeats: int,
    metadata: dict,
    record_nvidia_smi: bool,
) -> dict:
    if hasattr(jax, "clear_caches"):
        jax.clear_caches()

    nvidia_smi_before = nvidia_smi_snapshot(record_nvidia_smi)
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

    run_fn = lambda: beamformer(**setup.data)
    first_run, first_result = time_call(run_fn)
    rss_after_first = rss_gb()
    run_warmups(run_fn, warmups)
    rss_after_warmups = rss_gb()
    repeat_times, repeat_result = timed_repeats(run_fn, repeats)
    rss_after_repeats = rss_gb()
    result = repeat_result if repeat_result is not None else first_result
    result.block_until_ready()

    timing = compute_timing_stats(repeat_times)
    nvidia_smi_after = nvidia_smi_snapshot(record_nvidia_smi)

    return {
        **metadata,
        "nvidia_smi_before_case": nvidia_smi_before,
        "case": case.name,
        "strategy": strategy_name,
        "scan": case.scan_label,
        "points": sizes["points"],
        "receivers": sizes["receivers"],
        "transmits": sizes["transmits"],
        "conceptual_cube_gb": conceptual_cube_gb,
        "warmups": warmups,
        "setup_s": setup_time,
        "build_s": build_time,
        "first_run_s": first_run,
        "second_run_s": first_or_empty(repeat_times, 0),
        "third_run_s": first_or_empty(repeat_times, 1),
        **timing,
        "rss_before_build_gb": rss_before_build,
        "rss_after_build_gb": rss_after_build,
        "rss_after_first_gb": rss_after_first,
        "rss_after_warmups_gb": rss_after_warmups,
        "rss_after_repeats_gb": rss_after_repeats,
        "nvidia_smi_after_case": nvidia_smi_after,
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
        "steady_median_s",
        "steady_mean_s",
        "steady_min_s",
        "steady_max_s",
        "rss_after_repeats_gb",
        "result_shape",
    ]
    widths = {
        header: max(
            len(header),
            *[
                len(f"{row[header]:.4f}")
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
                value = f"{value:.4f}"
            values.append(str(value).ljust(widths[header]))
        print(" ".join(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--native",
        action="store_true",
        help="Also benchmark native scan with 75 transmits.",
    )
    parser.add_argument(
        "--cases",
        help=(
            "Comma-separated case names to run. Use 'all' for every case. "
            "Available: " + ", ".join(case.name for case in ALL_CASES)
        ),
    )
    parser.add_argument(
        "--strategies",
        help=(
            "Comma-separated strategy names to run. Use 'all' for every strategy. "
            "Available: " + ", ".join(name for name, _ in STRATEGIES)
        ),
    )
    parser.add_argument("--warmups", type=nonnegative_int, default=0)
    parser.add_argument("--repeats", type=positive_int, default=2)
    parser.add_argument("--data-path", type=Path, help="Explicit PICMUS UFF file path.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=SCRIPT_DIR / "data",
        help="Cache directory used when --data-path is not provided.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "outputs",
        help="Directory for the CSV output.",
    )
    parser.add_argument(
        "--record-nvidia-smi",
        action="store_true",
        help="Record nvidia-smi snapshots before and after each case.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backend_manager.active_backend = "jax"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    default_cases = ALL_CASES if args.native else [ALL_CASES[0]]
    cases = select_cases(ALL_CASES if args.cases else default_cases, args.cases)
    strategies = select_names(STRATEGIES, args.strategies, "strategy")

    print(f"Python executable: {sys.executable}")
    backend = jax.default_backend()
    print(f"JAX version: {jax.__version__}")
    print(f"JAX backend: {backend}")
    print(f"JAX devices: {jax.devices()}")
    print("Downloading/loading dataset...")
    uff_path, download_time = resolve_uff_path(
        data_path=args.data_path,
        data_dir=args.data_dir,
    )
    print(f"UFF path: {uff_path}")
    print(f"download/cache time: {download_time:.3f} s")

    print("Reading UFF channel_data and scan...")
    read_start = time.perf_counter()
    uff = Uff(str(uff_path))
    channel_data = uff.read("/channel_data")
    scan = uff.read("/scan")
    read_time = seconds_since(read_start)
    print(f"UFF read time: {read_time:.3f} s")
    print("raw channel_data.data shape:", getattr(channel_data.data, "shape", None))

    metadata = benchmark_metadata(jax, uff_path)
    metadata["uff_read_s"] = read_time
    metadata["download_cache_s"] = download_time

    rows = []
    for case in cases:
        for strategy_name, builder in strategies:
            print()
            print(
                f"Running case={case.name} strategy={strategy_name} "
                f"scan={case.scan_label} n_tx={case.n_tx}"
            )
            rows.append(
                run_strategy(
                    channel_data,
                    scan,
                    case,
                    strategy_name,
                    builder,
                    warmups=args.warmups,
                    repeats=args.repeats,
                    metadata=metadata,
                    record_nvidia_smi=args.record_nvidia_smi,
                )
            )

    print_table(rows)

    output_path = args.output_dir / f"picmus_transmit_strategy_benchmark_{backend}.csv"
    write_csv(output_path, rows)
    print()
    print(f"Saved CSV: {output_path}")


if __name__ == "__main__":
    main()




