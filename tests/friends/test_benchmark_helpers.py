from pathlib import Path

import pytest

from friends.benchmark_helpers import (
    BenchmarkCase,
    compute_timing_stats,
    resolve_uff_path,
    select_cases,
)


def test_select_cases_filters_by_name():
    cases = [
        BenchmarkCase("small", 4, 5, 1),
        BenchmarkCase("native", None, None, 75),
    ]

    selected = select_cases(cases, "native")

    assert [case.name for case in selected] == ["native"]


def test_select_cases_rejects_unknown_name():
    cases = [BenchmarkCase("small", 4, 5, 1)]

    with pytest.raises(ValueError, match="Unknown case"):
        select_cases(cases, "missing")


def test_compute_timing_stats_includes_median_and_p90():
    stats = compute_timing_stats([0.004, 0.001, 0.003, 0.002])

    assert stats["steady_mean_s"] == pytest.approx(0.0025)
    assert stats["steady_median_s"] == pytest.approx(0.0025)
    assert stats["steady_min_s"] == pytest.approx(0.001)
    assert stats["steady_max_s"] == pytest.approx(0.004)
    assert stats["steady_p90_s"] == pytest.approx(0.0037)
    assert stats["steady_repeats"] == 4


def test_resolve_uff_path_uses_explicit_data_path(tmp_path):
    uff = tmp_path / "PICMUS_carotid_cross.uff"
    uff.write_bytes(b"uff")

    path, elapsed = resolve_uff_path(data_path=uff, data_dir=tmp_path / "cache")

    assert path == uff
    assert elapsed == 0.0


def test_resolve_uff_path_rejects_missing_explicit_data_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_uff_path(data_path=tmp_path / "missing.uff", data_dir=tmp_path)
