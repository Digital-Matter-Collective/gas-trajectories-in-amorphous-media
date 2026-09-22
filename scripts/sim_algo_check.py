import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any, Callable, List, cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from base.bufferedsampler import BufferedSampler
from base.discretecdf import DiscreteCDF
from base.empiricalcdf import EmpiricalCDF
from base.trajectory import Trajectory
from base.trap_sequence import TrapSequence
from processes.distribution_fitter import GammaFitter, WeibullFitter
from processes.kerogen_walk_simulator import KerogenWalkSimulator
from processes.trajectory_analyzer.dm import DistanceMatrixAnalyzer
from processes.trajectory_analyzer.hybrid import HybridAnalyzer, HybridParams
from processes.trajectory_analyzer.np import (
    NeymanPearsonAnalyzer,
    NeymanPearsonParams,
)
from processes.trajectory_analyzer.sib import (
    StructureInformedBayesAnalyzer,
    StructureInformedBayesParams,
)
from processes.trap_extractor import TRAP_EXTRACTOR_VERSION, TrapExtractor
from scripts.dm_parameter_search import K_VALUES, optimized_dm_candidate_for_k
from utils.logging_setup import setup_logging
from utils.utils import create_empirical_cdf, kprint, ps_generate

DEFAULT_TRAJECTORY_COUNT = 100
DEFAULT_STEP_COUNT = 3000
DEFAULT_SEED = 42
CHECKPOINT_INTERVAL = 10
_MIGRATABLE_TRAP_EXTRACTOR_VERSION = 2

FIGURE_8_ANALYZERS = ("DM", "SIB", "HYB")
FIGURE_13_ANALYZERS = ("DM", "NP", "NP + Bayesian (SIB)")
ERROR_CORRIDOR_QUANTILES = (0.2, 0.8)
ERROR_CENTER = "mean"
ERROR_SERIES_COLORS = {
    "DM": "tab:blue",
    "SIB": "tab:orange",
    "NP + Bayesian (SIB)": "tab:orange",
    "HYB": "tab:green",
    "NP": "tab:red",
}

ErrorSummary = tuple[np.ndarray, float]
FigureErrorData = dict[str, ErrorSummary]


def _error_band(
    values: np.ndarray,
    *,
    q_low: float,
    q_high: float,
    center: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if values.ndim != 2:
        raise ValueError("Error data must have shape (probabilities, runs)")
    if center not in {"mean", "median"}:
        raise ValueError(f"Unsupported center statistic: {center}")
    if not 0 <= q_low <= q_high <= 1:
        raise ValueError("Expected 0 <= q_low <= q_high <= 1")

    central = (
        values.mean(axis=1)
        if center == "mean"
        else np.quantile(values, 0.5, axis=1)
    )
    low = np.quantile(values, q_low, axis=1)
    high = np.quantile(values, q_high, axis=1)
    # A mean can lie outside an equal-tail quantile interval for a skewed
    # distribution. Keep the requested percentile corridor while extending
    # its envelope just enough to include the displayed central statistic.
    low = np.minimum(low, central)
    high = np.maximum(high, central)
    return low, central, high


def _shared_error_axis_limits(
    data: FigureErrorData,
    *,
    q_low: float,
    q_high: float,
    center: str,
) -> tuple[float, float]:
    extrema: List[npt.NDArray[np.float64]] = []
    for errors, _ in data.values():
        low, central, high = _error_band(
            errors,
            q_low=q_low,
            q_high=q_high,
            center=center,
        )
        extrema.extend((low, central, high))

    minimum = min(float(values.min()) for values in extrema)
    maximum = max(float(values.max()) for values in extrema)
    if not np.isfinite(minimum) or not np.isfinite(maximum):
        raise ValueError("Cannot plot non-finite error statistics")

    span = maximum - minimum
    padding = 0.05 * span if span > 0 else 0.05 * max(abs(maximum), 1.0)
    lower = 0.0 if minimum >= 0 else minimum - padding
    return lower, maximum + padding


def _figure_error_data(
    dm_summary: ErrorSummary,
    np_summary: ErrorSummary,
    sib_summary: ErrorSummary,
    hybrid_summary: ErrorSummary,
) -> tuple[FigureErrorData, FigureErrorData]:
    figure_8_data = {
        "DM": dm_summary,
        "SIB": sib_summary,
        "HYB": hybrid_summary,
    }
    figure_13_data = {
        "DM": dm_summary,
        "NP": np_summary,
        "NP + Bayesian (SIB)": sib_summary,
    }
    return figure_8_data, figure_13_data


def save_error_corridor(
    out_dir: str | Path,
    filename: str,
    prob_grid: np.ndarray,
    data: FigureErrorData,
    title: str,
    q_low: float,
    q_high: float,
    center: str = "mean",
    y_limits: tuple[float, float] | None = None,
    legend_loc: str = "best",
) -> Path:
    """Save error curves with a quantile corridor."""
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    for label, (errors, k_est) in data.items():
        low, central, high = _error_band(
            errors,
            q_low=q_low,
            q_high=q_high,
            center=center,
        )
        color = ERROR_SERIES_COLORS[label]
        legend_label = f"{label}, " + str(r"$k_{est}=$") + f"{k_est:.3f}"
        ax.fill_between(prob_grid, low, high, color=color, alpha=0.2)
        ax.plot(prob_grid, central, color=color, label=legend_label)

    ax.set_xlabel(r"Return probability, $p$", fontsize=12)
    ax.set_ylabel(r"Average error, $E$", fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.tick_params(axis="both", labelsize=10)
    ax.legend(frameon=False, loc=legend_loc, prop={"size": 12})
    if y_limits is not None:
        ax.set_ylim(y_limits)

    output_path = output_dir / filename
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _atomic_savez(path: str | Path, **arrays: Any) -> None:
    """Atomically replace an .npz without writing through the target file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    with temporary.open("wb") as file:
        np.savez(file, **arrays)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, target)


def _save_trajectory_cache(
    path: Path, metadata: dict[str, object], trajectories: List[Trajectory]
) -> None:
    arrays: dict[str, Any] = {
        "metadata_json": json.dumps(metadata),
        "count": len(trajectories),
    }
    for index, trj in enumerate(trajectories):
        arrays.update(trj.to_npz_arrays(f"traj_{index}"))
    _atomic_savez(path, **arrays)


def _load_trajectory_cache(
    path: Path,
) -> tuple[dict[str, object], List[Trajectory]] | None:
    with np.load(path) as data:
        if "metadata_json" not in data:
            return None
        metadata = json.loads(str(data["metadata_json"]))
        count = int(data["count"])
        trajectories = [
            Trajectory.from_npz_arrays(f"traj_{index}", data)
            for index in range(count)
        ]
    return metadata, trajectories


def _atomic_json_dump(obj: Any, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(obj, file, ensure_ascii=False, indent=2)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, target)


def trajectory_seed(
    base_seed: int,
    k_index: int,
    p_index: int,
    trajectory_index: int,
) -> int:
    """Derive an independent reproducible seed for one trajectory."""
    seed_sequence = np.random.SeedSequence(
        [base_seed, k_index, p_index, trajectory_index]
    )
    return int(seed_sequence.generate_state(1, dtype=np.uint32)[0])


def estimate_trapping_probability(sequence: TrapSequence) -> float:
    """Return k_est = N_t / (N_0 + N_t) for one classified trajectory."""
    n_zero = int(sequence.get_zero_trap_count())
    n_trapped = int(sequence.get_non_zero_trap_count())
    event_count = n_zero + n_trapped
    if event_count == 0:
        raise ValueError(
            "Cannot estimate k from a sequence without trap events"
        )
    return n_trapped / event_count


def _build_simulator(
    radiuses: np.ndarray,
    throat_fitter: Any,
    step_count_distribution: DiscreteCDF,
    *,
    k: float,
    p: float,
    seed: int,
) -> KerogenWalkSimulator:
    # KerogenWalkSimulator currently draws from both global RNG modules.
    np.random.seed(seed)
    random.seed(seed)
    pore_size_distribution = create_empirical_cdf(radiuses)
    pore_sizes = BufferedSampler(
        EmpiricalCDF(pore_size_distribution), "psd", size=10_000
    )
    throat_lengths = BufferedSampler(throat_fitter, "ptl", size=10_000)
    step_counts = BufferedSampler(step_count_distribution, "ps", size=10_000)
    return KerogenWalkSimulator(
        pore_sizes,
        step_counts,
        throat_lengths,
        k,
        p,
    )


def _trajectory_cache_metadata(
    *,
    base_seed: int,
    k: float,
    p: float,
    k_index: int,
    p_index: int,
    trajectory_count: int,
    step_count: int,
) -> dict[str, object]:
    return {
        "base_seed": base_seed,
        "k": k,
        "p": p,
        "k_index": k_index,
        "p_index": p_index,
        "trajectory_count": trajectory_count,
        "step_count": step_count,
        "trajectory_seeds": [
            trajectory_seed(base_seed, k_index, p_index, index)
            for index in range(trajectory_count)
        ],
    }


def trajectories_simulation(
    output_dir: str | Path,
    *,
    trajectory_count: int,
    k: float,
    k_index: int,
    prob_grid: np.ndarray,
    step_count: int,
    radiuses: np.ndarray,
    throat_fitter: Any,
    step_count_distribution: DiscreteCDF,
    base_seed: int,
    force_recompute: bool,
) -> dict[tuple[float, float], List[Trajectory]]:
    """Generate or resume independently seeded synthetic trajectories."""
    trajectory_dir = Path(output_dir) / "trajectories"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    trajectories = {}

    for p_index, p_value in enumerate(prob_grid):
        p = float(p_value)
        cache_path = trajectory_dir / f"k={k:.2f}_p={p:.2f}.npz"
        metadata = _trajectory_cache_metadata(
            base_seed=base_seed,
            k=k,
            p=p,
            k_index=k_index,
            p_index=p_index,
            trajectory_count=trajectory_count,
            step_count=step_count,
        )

        cached_trajectories: List[Trajectory] = []
        if cache_path.is_file() and not force_recompute:
            loaded = _load_trajectory_cache(cache_path)
            if loaded is None:
                raise RuntimeError(
                    f"Legacy trajectory cache without seed metadata: {cache_path}. "
                    "Rerun once with --force-recompute."
                )
            cached_metadata, cached_trajectories = loaded
            if cached_metadata != metadata:
                raise RuntimeError(
                    f"Trajectory cache metadata mismatch: {cache_path}. "
                    "Use the original parameters or --force-recompute."
                )
            if len(cached_trajectories) > trajectory_count:
                raise RuntimeError(
                    f"Too many trajectories in cache: {cache_path}"
                )
        else:
            _save_trajectory_cache(cache_path, metadata, [])

        seeds = cast(List[int], metadata["trajectory_seeds"])
        for trajectory_index in range(
            len(cached_trajectories), trajectory_count
        ):
            simulator = _build_simulator(
                radiuses,
                throat_fitter,
                step_count_distribution,
                k=k,
                p=p,
                seed=int(seeds[trajectory_index]),
            )
            cached_trajectories.append(simulator.run(step_count + 1))
            _save_trajectory_cache(cache_path, metadata, cached_trajectories)

        trajectories[(k, p)] = cached_trajectories
        kprint(f"Trajectories ready: k={k}, p={p}, count={trajectory_count}")
    return trajectories


def _checkpoint_metadata(
    *,
    analyzer_name: str,
    k: float,
    prob_grid: np.ndarray,
    trajectory_count: int,
    step_count: int,
    seed: int,
) -> dict[str, object]:
    return {
        "analyzer": analyzer_name,
        "k": k,
        "prob_grid": [float(value) for value in prob_grid],
        "trajectory_count": trajectory_count,
        "step_count": step_count,
        "seed": seed,
        "error_metric": "relative_classification_error",
        "k_est_definition": "N_t / (N_0 + N_t)",
        "trap_extractor_version": TRAP_EXTRACTOR_VERSION,
    }


def _empty_checkpoint(
    result_shape: tuple[int, int], step_count: int
) -> dict[str, Any]:
    return {
        "next_flat_index": 0,
        "k_est": np.full(result_shape, np.nan, dtype=np.float64),
        "errors": np.full(result_shape, np.nan, dtype=np.float64),
        "results": np.full((*result_shape, step_count), -1, dtype=np.int8),
    }


def _validate_checkpoint_state(
    state: dict[str, Any],
    result_shape: tuple[int, int],
    step_count: int,
) -> None:
    total = int(np.prod(result_shape))
    next_flat_index = state["next_flat_index"]
    if (
        not isinstance(next_flat_index, int)
        or not 0 <= next_flat_index <= total
    ):
        raise RuntimeError(f"Invalid checkpoint index: {next_flat_index}")
    if state["k_est"].shape != result_shape:
        raise RuntimeError("Invalid k_est checkpoint shape")
    if state["errors"].shape != result_shape:
        raise RuntimeError("Invalid errors checkpoint shape")
    if state["results"].shape != (*result_shape, step_count):
        raise RuntimeError("Invalid classification-results checkpoint shape")

    completed_k = state["k_est"].reshape(-1)[:next_flat_index]
    completed_errors = state["errors"].reshape(-1)[:next_flat_index]
    completed_results = state["results"].reshape(total, step_count)[
        :next_flat_index
    ]
    if not np.all(np.isfinite(completed_k)):
        raise RuntimeError(
            "Completed checkpoint prefix has missing k_est values"
        )
    if not np.all(np.isfinite(completed_errors)):
        raise RuntimeError("Completed checkpoint prefix has missing errors")
    if np.any(completed_results < 0):
        raise RuntimeError("Completed checkpoint prefix has missing labels")


def _save_analyzer_checkpoint(
    checkpoint_path: Path,
    metadata: dict[str, object],
    state: dict[str, Any],
) -> None:
    _atomic_savez(
        checkpoint_path,
        metadata_json=json.dumps(metadata),
        next_flat_index=state["next_flat_index"],
        k_est=state["k_est"],
        errors=state["errors"],
        results=state["results"],
    )


def _load_or_initialize_checkpoint(
    checkpoint_path: Path,
    metadata: dict[str, object],
    result_shape: tuple[int, int],
    step_count: int,
    *,
    force_recompute: bool,
) -> dict[str, Any]:
    if force_recompute or not checkpoint_path.is_file():
        state = _empty_checkpoint(result_shape, step_count)
        _save_analyzer_checkpoint(checkpoint_path, metadata, state)
        return state

    with np.load(checkpoint_path) as payload:
        if "metadata_json" not in payload:
            raise RuntimeError(
                f"Checkpoint metadata mismatch or legacy checkpoint: "
                f"{checkpoint_path}. Rerun once with --force-recompute."
            )
        recorded_metadata = json.loads(str(payload["metadata_json"]))
        version_2_metadata = {
            **metadata,
            "trap_extractor_version": _MIGRATABLE_TRAP_EXTRACTOR_VERSION,
        }
        unversioned_metadata = {
            key: value
            for key, value in metadata.items()
            if key != "trap_extractor_version"
        }
        migrate_k_est = recorded_metadata in (
            version_2_metadata,
            unversioned_metadata,
        )
        if recorded_metadata != metadata and not migrate_k_est:
            raise RuntimeError(
                f"Checkpoint metadata mismatch or legacy checkpoint: "
                f"{checkpoint_path}. Rerun once with --force-recompute."
            )

        state = {
            "next_flat_index": int(payload["next_flat_index"]),
            "k_est": np.asarray(payload["k_est"], dtype=np.float64),
            "errors": np.asarray(payload["errors"], dtype=np.float64),
            "results": np.asarray(payload["results"], dtype=np.int8),
        }
    _validate_checkpoint_state(state, result_shape, step_count)
    if migrate_k_est:
        completed_results = state["results"].reshape(-1, step_count)[
            : state["next_flat_index"]
        ]
        completed_k_est = state["k_est"].reshape(-1)
        for index, result in enumerate(completed_results):
            sequence = TrapExtractor.get_trap_seq(
                result.astype(np.bool_), delta_time=1.0
            )
            completed_k_est[index] = estimate_trapping_probability(sequence)
        _save_analyzer_checkpoint(checkpoint_path, metadata, state)
        kprint(
            f"Migrated {checkpoint_path}: recomputed k_est from cached "
            "classification labels without rerunning the analyzer"
        )
    return state


CALIBRATION_INPUT_FILES = (
    "pi_l_gamma_fitter.json",
    "throat_lengths_weibull_fitter.json",
    "radiuses.npy",
)


def _compute_code_version() -> str:
    """Return the current commit hash of the repository producing this data."""
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        check=True,
        text=True,
    )
    return result.stdout.strip()


def _compute_input_fingerprint(main_path: Path) -> dict[str, str]:
    """Hash each calibration input feeding the synthetic benchmark."""
    fingerprint: dict[str, str] = {}
    for filename in CALIBRATION_INPUT_FILES:
        digest = hashlib.sha256()
        with (main_path / filename).open("rb") as file:
            for chunk in iter(lambda: file.read(1 << 20), b""):
                digest.update(chunk)
        fingerprint[filename] = digest.hexdigest()
    return fingerprint


def _benchmark_manifest(
    *,
    seed: int,
    trajectory_count: int,
    step_count: int,
    prob_grid: np.ndarray,
    code_version: str,
    input_fingerprint: dict[str, str],
) -> dict[str, object]:
    return {
        "seed": seed,
        "trajectory_count": trajectory_count,
        "step_count": step_count,
        "k_values": list(K_VALUES),
        "p_values": [float(value) for value in prob_grid],
        "k_est_definition": "N_t / (N_0 + N_t)",
        "trap_extractor_version": TRAP_EXTRACTOR_VERSION,
        "bypass_event_definition": (
            "Each adjacent pair of inter-trap edges represents one fully "
            "observed zero-duration trap visit; trajectory boundaries are "
            "censored."
        ),
        "error_metric": "relative_classification_error",
        "figure_8": {
            "algorithms": list(FIGURE_8_ANALYZERS),
            "center": ERROR_CENTER,
            "corridor": "20th-80th percentile envelope including the mean",
        },
        "figure_13": {
            "algorithms": list(FIGURE_13_ANALYZERS),
            "center": ERROR_CENTER,
            "corridor": "20th-80th percentile envelope including the mean",
        },
        "code_version": code_version,
        "input_fingerprint": input_fingerprint,
        "notes": (
            "SIB is implemented as NP initialization followed by Bayesian "
            "refinement."
        ),
    }


def refresh_benchmark_manifest(path_to_main: str | Path) -> Path:
    """Recompute code_version/input_fingerprint in an existing manifest.

    Reuses seed/trajectory_count/step_count/p_values already recorded by a
    prior run, so this never re-runs the (expensive) synthetic benchmark and
    never touches any previously computed result.
    """
    main_path = Path(path_to_main)
    manifest_path = main_path / "errors" / "synthetic_benchmark_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as file:
        existing = json.load(file)

    manifest = _benchmark_manifest(
        seed=existing["seed"],
        trajectory_count=existing["trajectory_count"],
        step_count=existing["step_count"],
        prob_grid=np.asarray(existing["p_values"]),
        code_version=_compute_code_version(),
        input_fingerprint=_compute_input_fingerprint(main_path),
    )
    _atomic_json_dump(manifest, manifest_path)
    return manifest_path


def _save_synthetic_k_est_summary(
    rows: list[dict[str, object]], output_dir: Path
) -> tuple[Path, Path]:
    """Save the estimated capture probabilities for synthetic trajectories."""
    csv_path = output_dir / "synthetic_k_est_summary.csv"
    json_path = output_dir / "synthetic_k_est_summary.json"
    csv_temporary = csv_path.with_name(f".{csv_path.name}.tmp")
    fieldnames = [
        "algorithm",
        "k",
        "k_est",
        "relative_deviation_percent",
        "trajectory_count_per_p",
        "step_count",
        "seed",
        "p_values",
        "aggregation",
    ]
    with csv_temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = row.copy()
            csv_row["p_values"] = json.dumps(
                csv_row["p_values"], separators=(",", ":")
            )
            writer.writerow(csv_row)
        file.flush()
        os.fsync(file.fileno())
    os.replace(csv_temporary, csv_path)
    _atomic_json_dump(rows, json_path)
    return csv_path, json_path


def run(
    path_to_main: str | Path,
    *,
    trajectory_count: int = DEFAULT_TRAJECTORY_COUNT,
    step_count: int = DEFAULT_STEP_COUNT,
    seed: int = DEFAULT_SEED,
    force_recompute: bool = False,
) -> None:
    if trajectory_count <= 0:
        raise ValueError("trajectory_count must be positive")
    if step_count < 9:
        raise ValueError(
            "step_count must be at least 9 (ten trajectory points)"
        )

    main_path = Path(path_to_main)
    errors_dir = main_path / "errors"
    checkpoints_dir = errors_dir / "checkpoints"
    figures_dir = main_path / "figs"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    pi_l_path = main_path / "pi_l_gamma_fitter.json"
    throat_path = main_path / "throat_lengths_weibull_fitter.json"
    radiuses_path = main_path / "radiuses.npy"
    if not pi_l_path.is_file():
        raise RuntimeError(f"pi_l data not found: {pi_l_path}")
    if not throat_path.is_file():
        raise RuntimeError(f"throat_lengths not found: {throat_path}")
    if not radiuses_path.is_file():
        raise RuntimeError(f"radiuses not found: {radiuses_path}")

    with pi_l_path.open() as file:
        pi_l_fitter = GammaFitter.from_dict(json.load(file))
    with throat_path.open() as file:
        throat_fitter = WeibullFitter.from_dict(json.load(file))
    radiuses = np.load(radiuses_path)

    prob_grid = np.arange(0.0, 1.05, 0.05)
    step_count_distribution = DiscreteCDF(
        ps_generate("uniform", mean_count=100)
    )
    _atomic_json_dump(
        _benchmark_manifest(
            seed=seed,
            trajectory_count=trajectory_count,
            step_count=step_count,
            prob_grid=prob_grid,
            code_version=_compute_code_version(),
            input_fingerprint=_compute_input_fingerprint(main_path),
        ),
        errors_dir / "synthetic_benchmark_manifest.json",
    )

    probabilistic_params = {
        k: StructureInformedBayesParams(1e-3, 0.01) for k in K_VALUES
    }
    hybrid_params = {
        k: HybridParams(
            probabilistic_params[k],
            optimized_dm_candidate_for_k(k).to_params(),
            0.3,
        )
        for k in K_VALUES
    }
    np_params = {k: NeymanPearsonParams(0.01) for k in K_VALUES}
    sib_params = probabilistic_params
    summary_rows: list[dict[str, object]] = []

    for k_index, k in enumerate(K_VALUES):
        result_shape = (len(prob_grid), trajectory_count)
        analyzers: list[tuple[Any, Callable[[Any, int, int], None]]] = []

        matrix_analyzer = DistanceMatrixAnalyzer(
            optimized_dm_candidate_for_k(k).to_params()
        )
        np_analyzer = NeymanPearsonAnalyzer(
            np_params[k], pi_l_fitter, throat_fitter
        )
        sib_analyzer = StructureInformedBayesAnalyzer(
            sib_params[k], pi_l_fitter, throat_fitter
        )
        hybrid_analyzer = HybridAnalyzer(
            hybrid_params[k], pi_l_fitter, throat_fitter
        )

        trajectories = trajectories_simulation(
            errors_dir,
            trajectory_count=trajectory_count,
            k=k,
            k_index=k_index,
            prob_grid=prob_grid,
            step_count=step_count,
            radiuses=radiuses,
            throat_fitter=throat_fitter,
            step_count_distribution=step_count_distribution,
            base_seed=seed,
            force_recompute=force_recompute,
        )

        current_state: dict[str, dict[str, Any]] = {}

        def no_approximation(
            analyzer: Any, p_index: int, trj_index: int
        ) -> None:
            del analyzer, p_index, trj_index

        def use_dm_approximation(
            analyzer: HybridAnalyzer, p_index: int, trj_index: int
        ) -> None:
            dm_results = current_state[DistanceMatrixAnalyzer.name()]["results"]
            labels = dm_results[p_index, trj_index]
            if np.any(labels < 0):
                raise RuntimeError("HYB requested an incomplete DM result")
            analyzer.set_trap_approx(labels.astype(np.bool_))

        analyzers.extend(
            [
                (matrix_analyzer, no_approximation),
                (np_analyzer, no_approximation),
                (sib_analyzer, no_approximation),
                (hybrid_analyzer, use_dm_approximation),
            ]
        )

        for analyzer, approximation in analyzers:
            analyzer_name = analyzer.name()
            checkpoint_path = checkpoints_dir / (
                f"name={analyzer_name}_k={k}_count_trj={trajectory_count}_"
                f"count_steps={step_count}.npz"
            )
            metadata = _checkpoint_metadata(
                analyzer_name=analyzer_name,
                k=k,
                prob_grid=prob_grid,
                trajectory_count=trajectory_count,
                step_count=step_count,
                seed=seed,
            )
            state = _load_or_initialize_checkpoint(
                checkpoint_path,
                metadata,
                result_shape,
                step_count,
                force_recompute=force_recompute,
            )
            next_flat_index = state["next_flat_index"]
            total = len(prob_grid) * trajectory_count
            if next_flat_index:
                kprint(
                    f"Resume {analyzer_name} for k={k}: "
                    f"{next_flat_index}/{total} trajectories"
                )

            for flat_index in range(next_flat_index, total):
                p_index = flat_index // trajectory_count
                trajectory_index = flat_index % trajectory_count
                p = float(prob_grid[p_index])
                trajectory = trajectories[(k, p)][trajectory_index]

                approximation(analyzer, p_index, trajectory_index)
                result = analyzer.run(trajectory).astype(np.bool_)
                assert trajectory.traps is not None
                expected = trajectory.traps.astype(np.bool_)
                if result.shape != expected.shape:
                    raise RuntimeError(
                        f"{analyzer_name} result shape {result.shape} does not "
                        f"match ground truth {expected.shape}"
                    )

                delta_time = trajectory.delta_time * 1e-12
                sequence = TrapExtractor.get_trap_seq(result, delta_time)
                state["errors"][p_index, trajectory_index] = np.mean(
                    result != expected
                )
                state["k_est"][p_index, trajectory_index] = (
                    estimate_trapping_probability(sequence)
                )
                state["results"][p_index, trajectory_index] = result
                state["next_flat_index"] = flat_index + 1

                if (
                    state["next_flat_index"] % CHECKPOINT_INTERVAL == 0
                    or state["next_flat_index"] == total
                ):
                    _save_analyzer_checkpoint(checkpoint_path, metadata, state)
                    kprint(
                        f"Checkpoint {analyzer_name}, k={k}: "
                        f"{state['next_flat_index']}/{total}"
                    )

            current_state[analyzer_name] = state

        def summarized(analyzer_name: str) -> tuple[np.ndarray, float]:
            state = current_state[analyzer_name]
            if not np.all(np.isfinite(state["errors"])):
                raise RuntimeError(f"Incomplete errors for {analyzer_name}")
            if not np.all(np.isfinite(state["k_est"])):
                raise RuntimeError(f"Incomplete k_est for {analyzer_name}")
            return state["errors"], float(np.mean(state["k_est"]))

        dm_summary = summarized(DistanceMatrixAnalyzer.name())
        np_summary = summarized(NeymanPearsonAnalyzer.name())
        sib_summary = summarized(StructureInformedBayesAnalyzer.name())
        hybrid_summary = summarized(HybridAnalyzer.name())
        title = str(r"Trapping probability, $k$=") + f"{k}"
        figure_8_data, figure_13_data = _figure_error_data(
            dm_summary,
            np_summary,
            sib_summary,
            hybrid_summary,
        )
        q_low, q_high = ERROR_CORRIDOR_QUANTILES
        shared_y_limits = _shared_error_axis_limits(
            {
                "DM": dm_summary,
                "SIB": sib_summary,
                "HYB": hybrid_summary,
                "NP": np_summary,
            },
            q_low=q_low,
            q_high=q_high,
            center=ERROR_CENTER,
        )

        figure_8_path = save_error_corridor(
            figures_dir,
            f"fig08_errors_k={k}_trj={trajectory_count}_steps={step_count}.svg",
            prob_grid,
            figure_8_data,
            title,
            q_low=q_low,
            q_high=q_high,
            center=ERROR_CENTER,
            y_limits=shared_y_limits,
        )
        figure_13_path = save_error_corridor(
            figures_dir,
            f"fig13_errors_k={k}_trj={trajectory_count}_steps={step_count}.svg",
            prob_grid,
            figure_13_data,
            title,
            q_low=q_low,
            q_high=q_high,
            center=ERROR_CENTER,
            y_limits=shared_y_limits,
        )
        kprint(f"Saved Fig. 8 panel: {figure_8_path}")
        kprint(f"Saved Fig. 13 panel: {figure_13_path}")
        kprint(
            f"For k={k}: SIB k_est={sib_summary[1]:.3f}; "
            f"HYB k_est={hybrid_summary[1]:.3f}; "
            f"DM k_est={dm_summary[1]:.3f}"
        )

        for algorithm, summary in (
            ("DM", dm_summary),
            ("SIB", sib_summary),
            ("HYB", hybrid_summary),
        ):
            k_est = summary[1]
            summary_rows.append(
                {
                    "algorithm": algorithm,
                    "k": k,
                    "k_est": k_est,
                    "relative_deviation_percent": 100.0 * (k_est - k) / k,
                    "trajectory_count_per_p": trajectory_count,
                    "step_count": step_count,
                    "seed": seed,
                    "p_values": [float(value) for value in prob_grid],
                    "aggregation": (
                        "mean of per-trajectory k_est over all p values"
                    ),
                }
            )

    summary_csv, summary_json = _save_synthetic_k_est_summary(
        summary_rows, errors_dir
    )
    kprint(f"Saved synthetic benchmark summary: {summary_csv}")
    kprint(f"Saved synthetic benchmark summary: {summary_json}")


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Reproduce synthetic benchmarks for Figures 8 and 13"
    )
    parser.add_argument("path", type=Path, help="Data directory")
    parser.add_argument(
        "--trajectory-count",
        type=int,
        default=DEFAULT_TRAJECTORY_COUNT,
    )
    parser.add_argument(
        "--step-count",
        type=int,
        default=DEFAULT_STEP_COUNT,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Replace trajectory caches and analyzer checkpoints",
    )
    parser.add_argument(
        "--refresh-manifest-only",
        action="store_true",
        help=(
            "Recompute code_version/input_fingerprint in the existing "
            "synthetic_benchmark_manifest.json and exit, without "
            "re-running the benchmark"
        ),
    )
    arguments = parser.parse_args()

    if arguments.refresh_manifest_only:
        refreshed_path = refresh_benchmark_manifest(arguments.path)
        kprint(f"Refreshed {refreshed_path}")
    else:
        run(
            arguments.path,
            trajectory_count=arguments.trajectory_count,
            step_count=arguments.step_count,
            seed=arguments.seed,
            force_recompute=arguments.force_recompute,
        )
