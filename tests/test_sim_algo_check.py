import json
from pathlib import Path
from typing import Any, Dict, cast

import numpy as np
import pytest

from base.trap_sequence import TrapSequence
from scripts.sim_algo_check import (
    ERROR_CORRIDOR_QUANTILES,
    ERROR_SERIES_COLORS,
    FIGURE_8_ANALYZERS,
    FIGURE_13_ANALYZERS,
    _atomic_savez,
    _benchmark_manifest,
    _checkpoint_metadata,
    _compute_code_version,
    _compute_input_fingerprint,
    _empty_checkpoint,
    _error_band,
    _figure_error_data,
    _load_or_initialize_checkpoint,
    _save_analyzer_checkpoint,
    _shared_error_axis_limits,
    estimate_trapping_probability,
    refresh_benchmark_manifest,
    trajectory_seed,
)


def test_k_est_is_capture_events_divided_by_all_events() -> None:
    sequence = TrapSequence(
        traps=np.ones(5, dtype=np.int32),
        times=np.array([0.0, 0.0, 1.0, 2.0, 3.0]),
    )

    assert estimate_trapping_probability(sequence) == pytest.approx(3 / 5)


def test_trajectory_seeds_are_stable_and_independent() -> None:
    seeds = {
        trajectory_seed(42, k_index, p_index, trajectory_index)
        for k_index in range(3)
        for p_index in range(2)
        for trajectory_index in range(5)
    }

    assert len(seeds) == 30
    assert trajectory_seed(42, 1, 2, 3) == trajectory_seed(42, 1, 2, 3)
    assert trajectory_seed(43, 1, 2, 3) != trajectory_seed(42, 1, 2, 3)


def test_atomic_savez_uses_a_separate_temporary_file(tmp_path: Path) -> None:
    target = tmp_path / "checkpoint.npz"

    _atomic_savez(target, value=np.array(1))
    _atomic_savez(target, value=np.array(2))

    with np.load(target) as data:
        assert int(data["value"]) == 2
    assert not (tmp_path / ".checkpoint.npz.tmp").exists()


def test_checkpoint_resumes_from_the_next_unprocessed_trajectory(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "checkpoint.npz"
    result_shape = (2, 10)
    step_count = 9
    prob_grid = np.array([0.0, 0.5])
    metadata = _checkpoint_metadata(
        analyzer_name="dm",
        k=0.1,
        prob_grid=prob_grid,
        trajectory_count=10,
        step_count=step_count,
        seed=42,
    )
    state = _empty_checkpoint(result_shape, step_count)
    state["k_est"].reshape(-1)[:10] = 0.5
    state["errors"].reshape(-1)[:10] = 0.1
    state["results"].reshape(-1, step_count)[:10] = 0
    state["next_flat_index"] = 10
    _save_analyzer_checkpoint(checkpoint_path, metadata, state)

    restored = _load_or_initialize_checkpoint(
        checkpoint_path,
        metadata,
        result_shape,
        step_count,
        force_recompute=False,
    )

    assert restored["next_flat_index"] == 10
    assert np.all(restored["results"].reshape(-1, step_count)[:10] == 0)


@pytest.mark.parametrize("old_version", [None, 2])
def test_checkpoint_migrates_k_est_without_rerunning_classification(
    tmp_path: Path,
    old_version: int | None,
) -> None:
    checkpoint_path = tmp_path / "checkpoint.npz"
    result_shape = (1, 1)
    step_count = 5
    metadata = _checkpoint_metadata(
        analyzer_name="dm",
        k=0.5,
        prob_grid=np.array([0.0]),
        trajectory_count=1,
        step_count=step_count,
        seed=42,
    )
    old_metadata = metadata.copy()
    if old_version is None:
        old_metadata.pop("trap_extractor_version")
    else:
        old_metadata["trap_extractor_version"] = old_version
    state = _empty_checkpoint(result_shape, step_count)
    state["k_est"][0, 0] = 0.123
    state["errors"][0, 0] = 0.1
    state["results"][0, 0] = np.array(
        [False, False, True, True, False], dtype=np.int8
    )
    state["next_flat_index"] = 1
    _save_analyzer_checkpoint(checkpoint_path, old_metadata, state)

    restored = _load_or_initialize_checkpoint(
        checkpoint_path,
        metadata,
        result_shape,
        step_count,
        force_recompute=False,
    )

    assert restored["k_est"][0, 0] == pytest.approx(0.5)
    with np.load(checkpoint_path) as migrated_payload:
        assert json.loads(str(migrated_payload["metadata_json"])) == metadata


def test_seed_and_figure_contract_are_saved_with_code_fingerprint() -> None:
    manifest = _benchmark_manifest(
        seed=42,
        trajectory_count=100,
        step_count=3000,
        prob_grid=np.array([0.0, 0.5, 1.0]),
        code_version="deadbeef",
        input_fingerprint={"radiuses.npy": "abc123"},
    )

    assert manifest["seed"] == 42
    assert manifest["k_est_definition"] == "N_t / (N_0 + N_t)"
    figure_8 = cast(Dict[str, Any], manifest["figure_8"])
    figure_13 = cast(Dict[str, Any], manifest["figure_13"])
    assert tuple(figure_8["algorithms"]) == FIGURE_8_ANALYZERS
    assert tuple(figure_13["algorithms"]) == FIGURE_13_ANALYZERS
    assert figure_8["center"] == "mean"
    assert figure_13["center"] == "mean"
    expected_corridor = "20th-80th percentile envelope including the mean"
    assert figure_8["corridor"] == expected_corridor
    assert figure_13["corridor"] == expected_corridor
    assert manifest["code_version"] == "deadbeef"
    assert manifest["input_fingerprint"] == {"radiuses.npy": "abc123"}
    assert "not recorded" not in cast(str, manifest["notes"])


def test_compute_input_fingerprint_hashes_each_calibration_file(
    tmp_path: Path,
) -> None:
    (tmp_path / "pi_l_gamma_fitter.json").write_bytes(b"pi_l")
    (tmp_path / "throat_lengths_weibull_fitter.json").write_bytes(b"throat")
    (tmp_path / "radiuses.npy").write_bytes(b"radiuses")

    fingerprint = _compute_input_fingerprint(tmp_path)

    assert set(fingerprint) == {
        "pi_l_gamma_fitter.json",
        "throat_lengths_weibull_fitter.json",
        "radiuses.npy",
    }
    assert all(len(value) == 64 for value in fingerprint.values())


def test_compute_code_version_returns_the_current_commit_hash() -> None:
    version = _compute_code_version()

    assert len(version) == 40
    assert all(character in "0123456789abcdef" for character in version)


def test_refresh_benchmark_manifest_preserves_recorded_results(
    tmp_path: Path,
) -> None:
    errors_dir = tmp_path / "errors"
    errors_dir.mkdir()
    original = _benchmark_manifest(
        seed=42,
        trajectory_count=100,
        step_count=3000,
        prob_grid=np.array([0.0, 0.5, 1.0]),
        code_version="stale",
        input_fingerprint={"radiuses.npy": "stale"},
    )
    manifest_path = errors_dir / "synthetic_benchmark_manifest.json"
    manifest_path.write_text(json.dumps(original), encoding="utf-8")

    (tmp_path / "pi_l_gamma_fitter.json").write_bytes(b"pi_l")
    (tmp_path / "throat_lengths_weibull_fitter.json").write_bytes(b"throat")
    (tmp_path / "radiuses.npy").write_bytes(b"radiuses")

    refreshed_path = refresh_benchmark_manifest(tmp_path)

    assert refreshed_path == manifest_path
    with manifest_path.open(encoding="utf-8") as file:
        refreshed = json.load(file)

    assert refreshed["seed"] == original["seed"]
    assert refreshed["trajectory_count"] == original["trajectory_count"]
    assert refreshed["step_count"] == original["step_count"]
    assert refreshed["p_values"] == original["p_values"]
    assert refreshed["figure_8"] == original["figure_8"]
    assert refreshed["figure_13"] == original["figure_13"]
    assert refreshed["code_version"] != "stale"
    assert refreshed["input_fingerprint"] != {"radiuses.npy": "stale"}


def test_figures_share_dm_and_sib_data_statistics_and_colors() -> None:
    dm = (np.arange(12, dtype=float).reshape(3, 4), 0.4)
    np_summary = (np.ones((3, 4), dtype=float), 0.5)
    sib = (np.full((3, 4), 2.0), 0.6)
    hybrid = (np.full((3, 4), 3.0), 0.7)

    figure_8, figure_13 = _figure_error_data(
        dm,
        np_summary,
        sib,
        hybrid,
    )

    assert figure_8["DM"] is figure_13["DM"]
    assert figure_8["SIB"] is figure_13["NP + Bayesian (SIB)"]
    assert ERROR_CORRIDOR_QUANTILES == (0.2, 0.8)
    assert set(figure_8) | set(figure_13) <= set(ERROR_SERIES_COLORS)
    assert (
        ERROR_SERIES_COLORS["SIB"] == ERROR_SERIES_COLORS["NP + Bayesian (SIB)"]
    )

    q_low, q_high = ERROR_CORRIDOR_QUANTILES
    np.testing.assert_array_equal(
        _error_band(
            figure_8["DM"][0],
            q_low=q_low,
            q_high=q_high,
            center="mean",
        ),
        _error_band(
            figure_13["DM"][0],
            q_low=q_low,
            q_high=q_high,
            center="mean",
        ),
    )
    np.testing.assert_array_equal(
        _error_band(
            figure_8["SIB"][0],
            q_low=q_low,
            q_high=q_high,
            center="mean",
        ),
        _error_band(
            figure_13["NP + Bayesian (SIB)"][0],
            q_low=q_low,
            q_high=q_high,
            center="mean",
        ),
    )


def test_shared_error_axis_includes_every_displayed_series() -> None:
    summaries = {
        "DM": (np.array([[0.1, 0.2, 0.3]]), 0.1),
        "SIB": (np.array([[0.0, 0.1, 0.2]]), 0.2),
        "HYB": (np.array([[0.2, 0.3, 0.4]]), 0.3),
        "NP": (np.array([[0.5, 0.6, 0.7]]), 0.4),
    }
    q_low, q_high = ERROR_CORRIDOR_QUANTILES

    lower, upper = _shared_error_axis_limits(
        summaries,
        q_low=q_low,
        q_high=q_high,
        center="mean",
    )

    assert lower == 0.0
    assert upper > 0.66


def test_mean_center_stays_inside_percentile_envelope_for_skewed_data() -> None:
    values = np.array([[0.0] * 9 + [1.0]])

    low, central, high = _error_band(
        values,
        q_low=0.2,
        q_high=0.8,
        center="mean",
    )

    assert low[0] <= central[0] <= high[0]
    assert central[0] == pytest.approx(values.mean(axis=1)[0])
    assert high[0] == pytest.approx(central[0])
