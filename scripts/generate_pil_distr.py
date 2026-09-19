import argparse
import json
from os import listdir
from os.path import isfile, join
from pathlib import Path
from typing import Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.signal import savgol_filter

from base.reader import Reader
from processes.distribution_fitter import (
    GammaFitter,
    WeibullFitter,
)
from processes.pil_distr_generator import PiLDistrGenerator
from utils.cache_manifest import check_cache, path_fingerprint, write_manifest
from utils.logging_setup import setup_logging
from utils.timer import Timer
from utils.utils import kprint


def plot_ar(ar: npt.NDArray[Any], maxes: Any, title: str, xlabel: str) -> None:
    maxes.plot(ar[:, 0], ar[:, 1], label=xlabel)
    maxes.set_title(title, fontsize=12)
    maxes.set_xlabel(xlabel, fontsize=12)
    maxes.tick_params(axis='x', labelsize=12)
    maxes.tick_params(axis='y', labelsize=12)


def plot_hist(
    data: npt.NDArray[Any],
    maxes: Any,
    title: str,
    xlabel: str,
    n: int = 50,
    smooth: bool = True,
) -> None:
    p, bb = np.histogram(data, bins=n)
    xdel = bb[1] - bb[0]
    x = bb[:-1] + xdel * 0.5
    pn = p / np.sum(p * xdel)
    if smooth:
        pn = savgol_filter(pn, 25, 3)
        pn[pn < 0] = 0
    maxes.plot(x, pn, label=xlabel)
    maxes.set_title(title, fontsize=12)
    maxes.set_xlabel(xlabel, fontsize=12)
    maxes.tick_params(axis='x', labelsize=12)
    maxes.tick_params(axis='y', labelsize=12)


def plot_distributions(path_to_pnms: str, path_to_save_pil: str) -> None:

    radiuses, throat_lengths = get_radiuses_lengths(path_to_pnms)

    pi_l = np.load(path_to_save_pil + "pi_l.npy")

    fig, axs = plt.subplots(1, 3)
    plot_ar(pi_l, axs[2], "PDF(Step size in the trap)", "Step size (nm)")
    plot_hist(
        radiuses, axs[0], "PDF(Pore size distribution)", "Pore diameter (nm)"
    )
    plot_hist(
        throat_lengths,
        axs[1],
        "PDF(Throat length distribution)",
        "Throat length (nm)",
    )
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.legend(frameon=False, prop={'size': 12})


def get_radiuses_lengths(
    path_to_pnms: str, radius_min: float = Reader.PNM_MIN_RADIUS_NM
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    onlyfiles = [
        f for f in listdir(path_to_pnms) if isfile(join(path_to_pnms, f))
    ]
    onlyfiles = [
        join(path_to_pnms, file[:-10]) for file in onlyfiles if "_link1" in file
    ]
    radiuses: npt.NDArray[np.float64] = np.array([])
    throat_lengths: npt.NDArray[np.float64] = np.array([])
    for f in onlyfiles:
        r, t = Reader.read_pnm_data(f, border=radius_min)
        radiuses = np.concatenate((radiuses, r))
        throat_lengths = np.concatenate((throat_lengths, t))

    return radiuses, throat_lengths


def save_distribution_figures(
    path_to_save: str,
    radiuses: npt.NDArray[np.float64],
    throat_lengths: npt.NDArray[np.float64],
) -> None:
    figs_dir = Path(path_to_save) / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    path_pi_l_gf = Path(join(path_to_save, "pi_l_gamma_fitter.json"))
    path_tl_wf = Path(join(path_to_save, "throat_lengths_weibull_fitter.json"))
    path_pi_l_data = Path(join(path_to_save, "pi_l_data.npy"))

    with open(path_pi_l_gf) as f:
        gfitter = GammaFitter.from_dict(json.load(f))
    with open(path_tl_wf) as f:
        wfitter = WeibullFitter.from_dict(json.load(f))
    pi_l_data = np.load(path_pi_l_data)

    def _plot_hist_and_fit(
        ax: Any, data: npt.NDArray[Any], fitter: Any, xlabel: str
    ) -> None:
        p, bb = np.histogram(data, bins=60)
        xdel = bb[1] - bb[0]
        x = bb[:-1] + xdel * 0.5
        pn = p / np.sum(p * xdel)
        ax.bar(x, pn, width=xdel, alpha=0.5, color="tab:blue", label="Data")
        x_fit = np.linspace(data.min(), data.max(), 300)
        ax.plot(x_fit, fitter.pdf(x_fit), color="tab:orange", lw=2, label="Fit")
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("PDF", fontsize=12)
        ax.tick_params(labelsize=12)
        ax.legend(frameon=False, fontsize=11)

    # Figure 1: throat length distribution
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    _plot_hist_and_fit(ax1, throat_lengths, wfitter, "Throat length (nm)")
    ax1.set_title("Throat length distribution", fontsize=13)
    fig1.tight_layout()
    for ext in ("svg", "png"):
        fig1.savefig(figs_dir / f"throat_length_distribution.{ext}", dpi=200)
    plt.close(fig1)

    # Figure 2: pi(l) step-size distribution + pore radii rug
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    _plot_hist_and_fit(ax2, pi_l_data, gfitter, "Step size (nm)")
    unique_r = np.unique(radiuses)
    ax2.plot(
        unique_r,
        np.zeros_like(unique_r),
        "|",
        ms=15,
        color="tab:green",
        alpha=0.7,
        label="Pore radii (nm)",
    )
    ax2.set_title("π(l) step-size distribution", fontsize=13)
    ax2.legend(frameon=False, fontsize=11)
    fig2.tight_layout()
    for ext in ("svg", "png"):
        fig2.savefig(figs_dir / f"pil_distribution.{ext}", dpi=200)
    plt.close(fig2)


def generate_pil_distribution(
    path_to_pnms: str, path_to_save: str, radius_min: float
) -> None:
    path_rads = Path(join(path_to_save, "radiuses.npy"))
    path_lens = Path(join(path_to_save, "throat_lengths.npy"))
    path_units = join(path_to_save, "pnm_distribution_units.json")

    pnm_metadata = {"pnm_dir": path_fingerprint(Path(path_to_pnms))}
    samples_status = (
        check_cache(path_rads, pnm_metadata) if isfile(path_lens) else "missing"
    )
    if samples_status == "match":
        radiuses = np.load(path_rads)
        throat_lengths = np.load(path_lens)
    elif samples_status == "legacy":
        kprint(
            f"Upgrading legacy cache {path_rads} to provenance-tracked "
            "format (trusted as-is, not recomputed)"
        )
        radiuses = np.load(path_rads)
        throat_lengths = np.load(path_lens)
        write_manifest(path_rads, pnm_metadata)
    else:
        if samples_status == "mismatch":
            kprint(
                f"Cache {path_rads} does not match current PNM directory; recomputing"
            )
        radiuses, throat_lengths = get_radiuses_lengths(
            path_to_pnms, radius_min
        )
        np.save(path_rads, radiuses)
        np.save(path_lens, throat_lengths)
        with open(path_units, "w") as f:
            json.dump({"length_unit": "nm"}, f)
        write_manifest(path_rads, pnm_metadata)

    generator = PiLDistrGenerator()

    timer = Timer()
    timer.start()
    path_pi_l_gf = Path(join(path_to_save, "pi_l_gamma_fitter.json"))
    path_pi_l_data = Path(join(path_to_save, "pi_l_data.npy"))
    pi_l_metadata = {**pnm_metadata, "radius_min": radius_min}
    gfitter_status = check_cache(path_pi_l_gf, pi_l_metadata)
    if gfitter_status == "match":
        kprint("Using cached gamma fitter")
    else:
        if gfitter_status == "legacy":
            kprint(
                f"Upgrading legacy cache {path_pi_l_gf} to provenance-tracked "
                "format (trusted as-is, not recomputed)"
            )
            write_manifest(path_pi_l_gf, pi_l_metadata)
        else:
            if gfitter_status == "mismatch":
                kprint(
                    f"Cache {path_pi_l_gf} does not match current inputs; recomputing"
                )
            data_status = check_cache(path_pi_l_data, pi_l_metadata)
            if data_status == "match":
                data = np.load(path_pi_l_data)
            elif data_status == "legacy":
                kprint(
                    f"Upgrading legacy cache {path_pi_l_data} to "
                    "provenance-tracked format (trusted as-is, not recomputed)"
                )
                data = np.load(path_pi_l_data)
                write_manifest(path_pi_l_data, pi_l_metadata)
            else:
                data = generator.gen_set(radiuses)
                np.save(path_pi_l_data, data)
                write_manifest(path_pi_l_data, pi_l_metadata)
            gfitter = GammaFitter()
            gfitter.fit(data)
            with open(path_pi_l_gf, "w") as f:
                json.dump(gfitter.to_dict(), f)
            write_manifest(path_pi_l_gf, pi_l_metadata)
    timer.stop("Gamma fitter")

    timer.start()
    path_tl_wf = Path(join(path_to_save, "throat_lengths_weibull_fitter.json"))
    tl_status = check_cache(path_tl_wf, pnm_metadata)
    if tl_status == "match":
        kprint("Using cached Weibull fitter")
    elif tl_status == "legacy":
        kprint(
            f"Upgrading legacy cache {path_tl_wf} to provenance-tracked "
            "format (trusted as-is, not recomputed)"
        )
        write_manifest(path_tl_wf, pnm_metadata)
    else:
        if tl_status == "mismatch":
            kprint(
                f"Cache {path_tl_wf} does not match current PNM directory; recomputing"
            )
        wfitter = WeibullFitter()
        wfitter.fit(throat_lengths)
        with open(path_tl_wf, "w") as f:
            json.dump(wfitter.to_dict(), f)
        write_manifest(path_tl_wf, pnm_metadata)
    timer.stop("Weibull fitter")

    save_distribution_figures(path_to_save, radiuses, throat_lengths)


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Generate PIL distributions from PNM data"
    )
    parser.add_argument("pnm_dir", type=Path, help="PNM directory (input)")
    parser.add_argument(
        "output_dir", type=Path, help="Output directory for fitter JSON files"
    )
    parser.add_argument(
        "--x-min",
        type=float,
        default=Reader.PNM_MIN_RADIUS_NM,
        help="Minimum pore radius in nm (default: 0.003)",
    )
    args = parser.parse_args()

    generate_pil_distribution(
        str(args.pnm_dir), str(args.output_dir), args.x_min
    )


if __name__ == '__main__':
    main()
