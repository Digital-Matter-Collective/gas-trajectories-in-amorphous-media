import argparse
import csv
import json
import os
import time
from dataclasses import asdict, dataclass, fields
from os.path import isfile, join
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.stats import linregress

from base.trajectory import Trajectory
from base.trap_sequence import TrapSequence
from processes.distribution_fitter import GammaFitter, WeibullFitter
from processes.trajectory_analyzer.dm import (
    DistanceMatrixAnalyzer,
    DistanceMatrixParams,
)
from processes.trajectory_analyzer.hybrid import (
    HybridAnalyzer,
    HybridParams,
)
from processes.trajectory_analyzer.sib import (
    StructureInformedBayesAnalyzer,
    StructureInformedBayesParams,
)
from processes.trap_extractor import TRAP_EXTRACTOR_VERSION, TrapExtractor
from utils.cache_manifest import check_cache, file_fingerprint, write_manifest
from utils.logging_setup import setup_logging
from utils.types import f32
from utils.utils import kprint

_TRAPPING_HISTOGRAM_BINS = 50
_LEGACY_COUPLED_TRAP_EXTRACTOR_VERSION = 2


@dataclass(frozen=True)
class TrapTimeFitResult:
    slope: float
    intercept: float
    r_value: float
    p_value: float
    std_err: float
    mu: float
    n_fit_bins: int


@dataclass(frozen=True)
class TrapEventSummary:
    n_t_mean: float
    n_0_mean: float
    k_est: float
    n_trajectories: int
    n_nonzero_events: int


@dataclass(frozen=True)
class TrappingSummaryRow:
    gas: str
    classifier: str
    mu: float
    n_t_mean: float
    n_0_mean: float
    k_est: float
    fit_t_min_s: float
    fit_t_max_s: float
    histogram_bins: int
    n_fit_bins: int
    n_trajectories: int
    n_nonzero_events: int
    loglog_slope: float
    fit_r_squared: float
    fit_p_value: float
    fit_std_err: float


def summarize_trap_events(
    trap_list: Sequence[TrapSequence],
) -> TrapEventSummary:
    """Calculate trapping-event statistics for a trajectory collection."""
    if not trap_list:
        raise ValueError("At least one trap sequence is required")

    n_0 = np.asarray(
        [seq.get_zero_trap_count() for seq in trap_list], dtype=np.float64
    )
    n_t = np.asarray(
        [seq.get_non_zero_trap_count() for seq in trap_list], dtype=np.float64
    )
    event_count = n_0 + n_t
    if np.any(event_count == 0):
        raise ValueError("Every trap sequence must contain at least one event")

    # Average the per-trajectory capture probability. This is not,
    # in general, equal to the ratio of the ensemble-mean event counts.
    k_est = np.mean(n_t / event_count)
    return TrapEventSummary(
        n_t_mean=float(np.mean(n_t)),
        n_0_mean=float(np.mean(n_0)),
        k_est=float(k_est),
        n_trajectories=len(trap_list),
        n_nonzero_events=int(np.sum(n_t)),
    )


def _trapping_sort_key(row: TrappingSummaryRow) -> tuple[int, int, str, str]:
    classifier_order = {"DM": 0, "SIB": 1, "HYB": 2}
    gas_order = {"CH4": 0, "H2": 1}
    return (
        classifier_order.get(row.classifier, len(classifier_order)),
        gas_order.get(row.gas, len(gas_order)),
        row.classifier,
        row.gas,
    )


def save_trapping_summary(
    rows: Sequence[TrappingSummaryRow], output_dir: Path
) -> tuple[Path, Path]:
    """Save trapping rows to CSV/JSON, preserving rows from other gases."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "trapping_summary.csv"
    json_path = output_dir / "trapping_summary.json"

    merged: Dict[tuple[str, str], TrappingSummaryRow] = {}
    if json_path.is_file():
        existing_data = json.loads(json_path.read_text(encoding="utf-8"))
        if not isinstance(existing_data, list):
            raise ValueError(f"Expected a JSON list in {json_path}")
        for item in existing_data:
            existing_row = TrappingSummaryRow(**item)
            merged[(existing_row.gas, existing_row.classifier)] = existing_row

    for row in rows:
        merged[(row.gas, row.classifier)] = row
    sorted_rows = sorted(merged.values(), key=_trapping_sort_key)
    serialized_rows = [asdict(row) for row in sorted_rows]

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[field.name for field in fields(TrappingSummaryRow)],
        )
        writer.writeheader()
        writer.writerows(serialized_rows)

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(serialized_rows, file, ensure_ascii=False, indent=2)
        file.write("\n")

    return csv_path, json_path


def plot_trapping_on_axis(
    ax: Any,
    times: np.ndarray,  # sec
    t_min: float,
    t_max: float,
    label: str,
) -> TrapTimeFitResult:
    fit_t_min_s = t_min
    fit_t_max_s = t_max
    t_min *= 1e6  # to us
    t_max *= 1e6  # to us
    times_us = times * 1e6  # to us
    Pt, bin_edges = np.histogram(
        times_us,
        bins=_TRAPPING_HISTOGRAM_BINS,
        density=True,
    )
    t = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    mask = (t >= t_min) & (t <= t_max) & (Pt > 0)
    t_part = t[mask]
    Pt_part = Pt[mask]

    log_t = np.log(t_part)
    log_S = np.log(Pt_part)

    if t_part.size < 2:
        raise ValueError(
            f"Need at least two populated histogram bins for {label} "
            f"in [{fit_t_min_s}, {fit_t_max_s}] s"
        )

    slope, intercept, r_value, p_value, std_err = linregress(log_t, log_S)

    # The manuscript defines the positive PDF tail exponent by P(t) ~ t^-mu.
    mu = -float(slope)

    kprint(f"{label} - mu: {mu:.3f}")

    # empirical
    points = ax.loglog(
        t,
        Pt,
        marker="o",
        linestyle="none",
        alpha=0.35,
    )
    color = points[0].get_color()

    # fit
    if np.isfinite(mu):
        t_line = np.logspace(
            np.log10(t_part.min()), np.log10(t_part.max()), 200
        )
        S_line = np.exp(intercept) * (t_line**slope)

        ax.loglog(
            t_line,
            S_line,
            linewidth=2,
            color=color,
            label=label,
        )
        # Подписываем только SIB и Distance-matrix
        if "SIB" in label:
            idx = int(0.72 * (len(t_line) - 1))
            x_txt = t_line[idx]
            y_txt = S_line[idx]

            # текст слева от линии
            ax.annotate(
                rf"$\sim t^{{{slope:.2f}}}$",
                xy=(x_txt, y_txt),
                xytext=(-35, 0),  # <-- сдвиг: влево и чуть вверх
                textcoords="offset points",
                color=color,
                fontsize=30,
                ha="right",
                va="bottom",
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.7,
                    pad=1.5,
                ),
            )

        elif "DM" in label:
            idx = int(0.62 * (len(t_line) - 1))
            x_txt = t_line[idx]
            y_txt = S_line[idx]

            # текст справа от линии
            ax.annotate(
                rf"$\sim t^{{{slope:.2f}}}$",
                xy=(x_txt, y_txt),
                xytext=(30, 0),  # <-- сдвиг: вправо и чуть вверх
                textcoords="offset points",
                color=color,
                fontsize=30,
                ha="left",
                va="bottom",
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.7,
                    pad=1.5,
                ),
            )

    return TrapTimeFitResult(
        slope=float(slope),
        intercept=float(intercept),
        r_value=float(r_value),
        p_value=float(p_value),
        std_err=float(std_err),
        mu=mu,
        n_fit_bins=int(t_part.size),
    )


def plot_trap_tim_distr(
    trap_list: list[TrapSequence],
    gas: str,
    prefix: str,
    t_min: float,
    t_max: float,
    ax1: Any,
) -> TrappingSummaryRow:
    event_summary = summarize_trap_events(trap_list)
    time_tuple = tuple(trap.times for trap in trap_list)
    time_trapings = np.concatenate(time_tuple)
    non_zero_tt = time_trapings[time_trapings != 0]

    kprint(f"{prefix} - N_0 mean: {event_summary.n_0_mean:.3f}")
    kprint(f"{prefix} - N_t mean: {event_summary.n_t_mean:.3f}")
    kprint(f"{prefix} - k_est: {event_summary.k_est:.3f}")

    fit = plot_trapping_on_axis(ax1, non_zero_tt, t_min, t_max, prefix)

    return TrappingSummaryRow(
        gas=gas,
        classifier=prefix,
        mu=fit.mu,
        n_t_mean=event_summary.n_t_mean,
        n_0_mean=event_summary.n_0_mean,
        k_est=event_summary.k_est,
        fit_t_min_s=t_min,
        fit_t_max_s=t_max,
        histogram_bins=_TRAPPING_HISTOGRAM_BINS,
        n_fit_bins=fit.n_fit_bins,
        n_trajectories=event_summary.n_trajectories,
        n_nonzero_events=event_summary.n_nonzero_events,
        loglog_slope=fit.slope,
        fit_r_squared=fit.r_value**2,
        fit_p_value=fit.p_value,
        fit_std_err=fit.std_err,
    )


def get_struct_params(gas: str) -> DistanceMatrixParams:
    lmu = np.array(
        [0.5, 1.0, 1.5, 2.0, 2.5, 3.0] if gas == "CH4" else [1.5, 2.0],
        dtype=f32,
    )
    return DistanceMatrixParams(
        traj_type='fBm',
        nu=0.1,
        diag_percentile=0,
        kernel_size=0,
        list_mu=lmu,
        p_value=0.9,
    )


def run(
    path_to_main: str,
    gas: str,
    step: int,
    t_min_max: Dict[str, Tuple[float, float]],
    ax1: Any,
    recompute_prefixes: Optional[Set[str]] = None,
) -> list[TrappingSummaryRow]:
    recompute_prefixes = recompute_prefixes or set()
    traj_path = join(path_to_main, "trj.gro")
    pts_trapping = join(path_to_main, "traps")
    path_to_pil_gf: str = join(path_to_main, "pi_l_gamma_fitter.json")
    path_to_tl_wf: str = join(
        path_to_main, "throat_lengths_weibull_fitter.json"
    )

    os.makedirs(pts_trapping, exist_ok=True)

    if isfile(path_to_pil_gf):
        with open(path_to_pil_gf) as f:
            pil_gamma_fitter = GammaFitter.from_dict(json.load(f))
    else:
        raise RuntimeError("pi_l data not found")

    if isfile(path_to_tl_wf):
        with open(path_to_tl_wf) as f:
            throat_lengths_weibull_fitter = WeibullFitter.from_dict(
                json.load(f)
            )
    else:
        raise RuntimeError("throat_lengths not found")

    trajectories = Trajectory.read_trajectories(traj_path)
    trajectories = trajectories[::step]

    struct_params = get_struct_params(gas)

    prob_np_params = StructureInformedBayesParams(1e-3, 1e-2)
    hybrid_analyzer = HybridAnalyzer(
        HybridParams(prob_np_params, struct_params, 0.1),
        pil_gamma_fitter,
        throat_lengths_weibull_fitter,
    )
    dm_analyzer = DistanceMatrixAnalyzer(struct_params)
    sib_analyzer = StructureInformedBayesAnalyzer(
        prob_np_params,
        pil_gamma_fitter,
        throat_lengths_weibull_fitter,
    )

    results: Dict[Tuple[str, int], npt.NDArray[np.bool_]] = {}
    summary_rows = []
    for analyzer, prefix in [
        (dm_analyzer, "DM"),
        (sib_analyzer, "SIB"),
        # (prob_analyzer, "Probabilistic"),
        (hybrid_analyzer, "HYB"),
    ]:
        cur_pts = join(pts_trapping, prefix)
        os.makedirs(cur_pts, exist_ok=True)

        trap_list = []

        for i, trj in enumerate(trajectories):
            seq_file = Path(join(cur_pts, f"seq_{step * i}.npz"))
            traps_file = Path(join(cur_pts, f"traps_{step * i}.npz"))

            analyzer_cache_metadata = {
                "gas": gas,
                "step": step,
                "trajectory_index": i,
                "prefix": prefix,
                "struct_params": struct_params,
                "prob_np_params": prob_np_params,
            }
            use_traps_cache = (
                traps_file.is_file() and prefix not in recompute_prefixes
            )
            traps_cache_status = (
                check_cache(traps_file, analyzer_cache_metadata)
                if use_traps_cache
                else "missing"
            )
            migrate_coupled_manifest = False
            if traps_cache_status == "mismatch":
                # Extractor v2 incorrectly coupled the derived TrapSequence
                # version to the expensive analyzer-output cache.  Accept
                # that exact manifest once, then rewrite it using only the
                # inputs that can actually change ``traps``.
                legacy_metadata = {
                    **analyzer_cache_metadata,
                    "trap_extractor_version": (
                        _LEGACY_COUPLED_TRAP_EXTRACTOR_VERSION
                    ),
                }
                if check_cache(traps_file, legacy_metadata) == "match":
                    migrate_coupled_manifest = True
                else:
                    use_traps_cache = False

            if not use_traps_cache:
                kprint(
                    f"Cache {traps_file} does not match current analyzer "
                    "parameters; recomputing"
                )
                if prefix == "HYB":
                    approx_traps = results[("DM", i)]
                    hybrid_analyzer.set_trap_approx(approx_traps)

                start_time = time.time()
                traps = analyzer.run(trj)
                kprint(
                    f"Analize trajectory {i} is ready for {prefix}! Time: {time.time() - start_time}"
                )
                with open(traps_file, 'wb') as handle:
                    np.savez(handle, traps=traps)
                write_manifest(traps_file, analyzer_cache_metadata)
            else:
                if traps_cache_status == "legacy":
                    kprint(
                        f"Upgrading legacy cache {traps_file} to "
                        "provenance-tracked format (trusted as-is, not recomputed)"
                    )
                with np.load(traps_file) as data:
                    traps = data["traps"]
                if traps_cache_status == "legacy" or migrate_coupled_manifest:
                    write_manifest(traps_file, analyzer_cache_metadata)

            sequence_cache_metadata = {
                "trap_extractor_version": TRAP_EXTRACTOR_VERSION,
                "delta_time_sec": trj.delta_time_sec,
                "traps_file": file_fingerprint(traps_file),
            }
            use_sequence_cache = (
                seq_file.is_file()
                and prefix not in recompute_prefixes
                and check_cache(seq_file, sequence_cache_metadata) == "match"
            )
            if use_sequence_cache:
                seq = TrapSequence.load_npz(seq_file)
            else:
                seq = TrapExtractor.get_trap_seq(traps, trj.delta_time_sec)
                seq.save_npz(seq_file)
                write_manifest(seq_file, sequence_cache_metadata)
            results[(prefix, i)] = np.copy(traps)

            trap_list.append(seq)
        t_min, t_max = t_min_max[prefix]
        summary_rows.append(
            plot_trap_tim_distr(
                trap_list,
                gas,
                prefix,
                t_min,
                t_max,
                ax1,
            )
        )

    return summary_rows


_DEFAULT_T_MIN_MAX: Dict[str, Dict[str, Tuple[float, float]]] = {
    "CH4": {
        "SIB": (1e-12, 3e-7),
        "HYB": (1e-12, 3e-7),
        "DM": (1e-12, 8e-7),
        "SIB_Neamann-Pearson": (1e-12, 3e-7),
    },
    "H2": {
        "SIB": (1e-12, 5e-8),
        "HYB": (1e-12, 5e-8),
        "DM": (1e-12, 1.5e-7),
        "SIB_Neamann-Pearson": (1e-12, 5e-8),
    },
}

if __name__ == '__main__':
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Trap time distribution builder"
    )
    parser.add_argument("path", type=Path, help="Data directory")
    parser.add_argument(
        "--label", type=str, required=True, help="Gas label (e.g. CH4, H2)"
    )
    parser.add_argument(
        "--num", type=int, default=1, help="Molecule step (default: 1)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output SVG path (default: <path>/traps/P(t)_loglog.svg)",
    )
    parser.add_argument(
        "--recompute",
        action="append",
        choices=("DM", "SIB", "HYB"),
        default=[],
        help="Recompute one analyzer and reuse other caches (repeatable)",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignore cached seq/traps files and recompute all analyzers",
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=None,
        help=(
            "Directory for trapping CSV/JSON (default: <path>/traps). "
            "Use the same directory for CH4 and H2 to merge both gases."
        ),
    )
    args = parser.parse_args()

    t_min_max = _DEFAULT_T_MIN_MAX.get(args.label, _DEFAULT_T_MIN_MAX["CH4"])
    recompute_prefixes = (
        {"DM", "SIB", "HYB"} if args.force_recompute else set(args.recompute)
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    summary_rows = run(
        str(args.path),
        args.label,
        args.num,
        t_min_max,
        ax,
        recompute_prefixes=recompute_prefixes,
    )
    summary_dir = args.summary_dir or args.path / "traps"
    csv_path, json_path = save_trapping_summary(summary_rows, summary_dir)
    kprint(f"Saved trapping summary: {csv_path}")
    kprint(f"Saved trapping summary: {json_path}")

    # ======================
    # Figure 1 — Survival
    # ======================
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$t, \mu s$", fontsize=20)
    ax.set_ylabel(r"$P(t)$", fontsize=20)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(frameon=False, fontsize=20)
    fig.tight_layout()
    out_path = (
        args.output if args.output else args.path / "traps" / "P(t)_loglog.svg"
    )
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
