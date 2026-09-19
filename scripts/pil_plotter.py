import argparse
import json
from os.path import isfile, join
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import PowerNorm
from matplotlib.ticker import MaxNLocator
from scipy.stats import exponweib

from base.reader import Reader
from processes.distribution_fitter import GammaFitter
from processes.pil_distr_generator import PiLDistrGenerator
from utils.logging_setup import setup_logging
from utils.utils import kprint


def _centers_to_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("Expected at least two grid centers")

    edges = np.empty(values.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (values[:-1] + values[1:])
    edges[0] = values[0] - 0.5 * (values[1] - values[0])
    edges[-1] = values[-1] + 0.5 * (values[-1] - values[-2])
    return edges


def plot_distributions(
    path_to_save: str,
    radius_min: float,
    r_points: int,
    r_step: int,
    l_bins: int,
    heatmap_mode: str,
    heatmap_interpolation: str,
    heatmap_dpi: int,
    fit_sample_stride: int = 10,
) -> None:
    path_units = join(path_to_save, "pnm_distribution_units.json")
    if not isfile(path_units):
        raise RuntimeError("PIL distribution cache must be regenerated in nm")
    with open(path_units) as f:
        units = json.load(f)
    if units.get("length_unit") != "nm":
        raise RuntimeError("PIL distribution cache must be regenerated in nm")

    path_rads = Path(join(path_to_save, "radiuses.npy"))

    radiuses = np.load(path_rads)
    radiuses = radiuses[radiuses > radius_min]
    kprint("Count radiuses: ", len(radiuses))

    rad_max = radiuses.max()
    kprint(f"Max radius: {rad_max}")
    kprint(f"Min radius: {radiuses.min()}")

    # --- fit P(r) ---
    # Millions of edges accumulate here; a uniform stride subsample keeps the
    # same distribution shape at a fraction of the fit cost. Configurable via
    # --fit-sample-stride (author decision 2026-09-02: keep the subsampling,
    # just make the stride an input instead of a hardcoded 10).
    if fit_sample_stride < 1:
        raise ValueError("fit_sample_stride must be a positive integer")
    params = exponweib.fit(radiuses[::fit_sample_stride])
    kprint("Fit radiuses params: ", params)
    r_fit = np.linspace(radius_min, rad_max, 300)
    pr_fit = exponweib.pdf(r_fit, *params)

    generator = PiLDistrGenerator()

    sample_rad, l_vals, pi_cond, pr_sample = (
        generator.get_conditional_curves_from_fit(
            params=params,
            r_min=radius_min,
            r_max=rad_max,
            r_points=r_points,
            step=r_step,
            cl=l_bins,
        )
    )

    kprint("sample_rad:", sample_rad.min(), sample_rad.max(), sample_rad.shape)
    kprint("l_vals:", l_vals.min(), l_vals.max(), l_vals.shape)
    kprint("pi_cond shape:", pi_cond.shape)

    positive = pi_cond[pi_cond > 0]
    kprint("pi_cond min positive:", positive.min())
    kprint("pi_cond max:", positive.max())
    kprint("pi_cond q95:", np.quantile(positive, 0.95))
    kprint("pi_cond q99:", np.quantile(positive, 0.99))
    kprint("pi_cond q999:", np.quantile(positive, 0.999))

    imax = np.unravel_index(np.argmax(pi_cond), pi_cond.shape)
    kprint("argmax pi_cond:", imax)

    # Если pi_cond имеет форму (len(sample_rad), len(l_vals))
    if pi_cond.shape == (len(sample_rad), len(l_vals)):
        integrals_l = np.trapezoid(pi_cond, x=l_vals, axis=1)
        kprint("max at r,l:", sample_rad[imax[0]], l_vals[imax[1]])

    # Если вдруг форма транспонированная
    elif pi_cond.shape == (len(l_vals), len(sample_rad)):
        integrals_l = np.trapezoid(pi_cond, x=l_vals, axis=0)
        kprint("max at l,r:", l_vals[imax[0]], sample_rad[imax[1]])

    else:
        raise ValueError("Unexpected pi_cond shape")

    kprint("Integral_l min:", integrals_l.min())
    kprint("Integral_l mean:", integrals_l.mean())
    kprint("Integral_l max:", integrals_l.max())
    kprint(
        "Integral_l q05/q50/q95:", np.quantile(integrals_l, [0.05, 0.5, 0.95])
    )

    dl = float(np.mean(np.diff(l_vals)))
    kprint("dl:", dl)
    kprint("max pi_cond * dl:", positive.max() * dl)

    # These fit/summary numbers were previously only ever visible via
    # kprint; persist them next to the figures too.
    summary_path = Path(path_to_save) / "pil_distribution_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "radius_fit_params": [float(p) for p in params],
                "sample_rad_min": float(sample_rad.min()),
                "sample_rad_max": float(sample_rad.max()),
                "l_vals_min": float(l_vals.min()),
                "l_vals_max": float(l_vals.max()),
                "pi_cond_min_positive": float(positive.min()),
                "pi_cond_max": float(positive.max()),
                "pi_cond_q95": float(np.quantile(positive, 0.95)),
                "pi_cond_q99": float(np.quantile(positive, 0.99)),
                "pi_cond_q999": float(np.quantile(positive, 0.999)),
                "integral_l_min": float(integrals_l.min()),
                "integral_l_mean": float(integrals_l.mean()),
                "integral_l_max": float(integrals_l.max()),
                "integral_l_q05_q50_q95": [
                    float(v)
                    for v in np.quantile(integrals_l, [0.05, 0.5, 0.95])
                ],
                "dl": dl,
                "max_pi_cond_times_dl": float(positive.max() * dl),
            },
            f,
            indent=2,
        )
        f.write("\n")
    kprint(f"Saved distribution summary: {summary_path}")

    kprint("sample_rad.shape: ", sample_rad.shape)
    kprint("l_vals.shape: ", l_vals.shape)
    kprint("pi_cond.shape: ", pi_cond.shape)
    kprint("pr_sample.shape: ", pr_sample.shape)

    # --- fit Π(l|r) ---

    path_pi_l_gf = Path(join(path_to_save, "pi_l_gamma_fitter.json"))
    if not path_pi_l_gf.exists():
        data = generator.gen_set(radiuses)
        gfitter = GammaFitter()
        gfitter.fit(data)
        with open(path_pi_l_gf, "w") as f:
            json.dump(gfitter.to_dict(), f)
    else:
        with open(path_pi_l_gf) as f:
            gfitter = GammaFitter.from_dict(json.load(f))
        kprint("Using cached gamma fitter")
    kprint("Finish fit")

    x = np.linspace(0, rad_max, 300)
    y_pl_fit = gfitter.pdf(x)

    # --- mask zeros in Π(l|r) so they are transparent ---
    pi_masked = np.ma.masked_where(pi_cond == 0, pi_cond)

    fig, ax = plt.subplots(figsize=(9, 7), facecolor="white")
    ax.set_facecolor("white")

    # --- upper panel in the same axes: heatmap Π(l|r), y >= 0 ---
    # cmap = plt.colormaps["magma"].copy()
    cmap = plt.colormaps["YlOrRd"].copy()
    cmap.set_bad(alpha=0)

    norm = PowerNorm(
        gamma=0.45,
        vmin=0.0,
        vmax=np.quantile(positive, 0.995),
        clip=True,
    )

    mesh: ScalarMappable
    if heatmap_mode == "mesh":
        mesh = ax.pcolormesh(
            sample_rad,
            l_vals,
            pi_masked.T,
            cmap=cmap,
            norm=norm,
            shading="auto",
        )
    elif heatmap_mode == "smooth":
        r_edges = _centers_to_edges(sample_rad)
        l_edges = _centers_to_edges(l_vals)
        mesh = ax.imshow(
            pi_masked.T,
            extent=(r_edges[0], r_edges[-1], l_edges[0], l_edges[-1]),
            origin="lower",
            aspect="auto",
            cmap=cmap,
            norm=norm,
            interpolation=heatmap_interpolation,
            resample=True,
        )
    else:
        raise ValueError(f"Unexpected heatmap mode: {heatmap_mode}")

    # --- lower schematic part: scaled P(r), y < 0 ---
    coeff = 0.5
    lower_height = coeff * float(l_vals.max())

    # общий масштаб для обеих плотностей
    global_max = max(pr_fit.max(), y_pl_fit.max())
    scale = lower_height / global_max

    y_pr = -scale * pr_fit
    y_pl_pr = -scale * y_pl_fit

    ax.fill_between(x, y_pl_pr, 0, color="tab:blue", alpha=0.25)
    ax.plot(
        x,
        y_pl_pr,
        label=r"$\Pi(l)$",
        linewidth=2.5,
        color="tab:blue",
    )

    ax.fill_between(r_fit, y_pr, 0.0, color="orange", alpha=0.35, linewidth=0)
    ax.plot(r_fit, y_pr, color="darkorange", linewidth=2.5, label=r"$P(r)$")

    # divider / x-axis at y = 0
    ax.axhline(0.0, color="black", linewidth=1.2)
    y_tick_text = -0.025 * float(l_vals.max())  # чуть ниже линии y=0
    tick_len = 0.01 * float(l_vals.max())
    y_xlabel = -0.10 * float(l_vals.max())

    # --- limits ---
    # x_start = 0.003
    x_start = 0.0
    ax.set_xlim(x_start, float(sample_rad.max()))
    # Author decision 2026-09-02: the 0.05 upper limit is a deliberate zoom
    # into the informative part of the heatmap, not an accidental clip.
    ax.set_ylim(float(y_pr.min()) * 1.15, 0.05)  # float(l_vals.max()) * 1.03)

    # --- x axis ---
    xticks = np.linspace(float(sample_rad.min()), float(sample_rad.max()), 5)
    ax.set_xticks(xticks)
    ax.set_xticklabels([])
    ax.set_xlabel("")
    ax.tick_params(axis="x", length=0)

    ax.text(
        float(sample_rad.max()),
        y_xlabel,
        r"$r, h$ (nm)",
        fontsize=24,
        ha="right",
        va="top",
    )

    for i, x in enumerate(xticks):
        shift = (xticks.max() - xticks.min()) / len(xticks) / 3
        if i == 0:
            x += shift
        elif i + 1 == len(xticks):
            x -= shift
        ax.text(
            x,
            y_tick_text,
            f"{x:.2f}",
            fontsize=16,
            ha="center",
            va="top",
        )
        ax.plot(
            [x, x],
            [0.0, -tick_len],
            color="black",
            linewidth=1.0,
        )

    # --- y axis: show ticks only for l > 0 ---
    yticks_top = np.linspace(0.01, float(l_vals.max()), 5)
    ax.set_yticks(yticks_top)
    ax.set_yticklabels([f"{y:.2f}" for y in yticks_top], fontsize=16)
    ax.set_ylabel(r"$l$ (nm)", fontsize=24)

    # --- clean spines ---
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # --- manual label for lower schematic P(r) part ---
    x_text = x_start - 0.035 * (
        float(sample_rad.max()) - float(sample_rad.min())
    )
    y_text = 0.55 * float(y_pr.min())

    ax.text(
        x_text,
        y_text,
        r"$Density$",
        fontsize=24,
        rotation=90,
        va="center",
        ha="center",
    )
    # --- legend
    ax.legend(
        loc="lower right",
        fontsize=24,
        frameon=False,
        fancybox=False,
        framealpha=0.9,
    )

    # --- colorbar ---
    cax = ax.inset_axes(
        (
            0.10,  # отступ слева, доля ширины графика
            0.90,  # положение снизу
            0.80,  # ширина
            0.035,  # высота
        ),
        zorder=10,
    )
    cbar = fig.colorbar(mesh, cax=cax, orientation="horizontal")

    cbar.set_label(
        r"$\Pi(l \mid r)$",
        fontsize=20,
        labelpad=10,
    )
    cbar.ax.xaxis.set_ticks_position("top")
    cbar.ax.xaxis.set_label_position("top")
    cbar.ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    cbar.locator = MaxNLocator(nbins=4)
    cbar.update_ticks()
    cbar.ax.tick_params(axis="x", labelsize=14, pad=1)

    fig.savefig(
        join(path_to_save, "figs", "pilr_2d.svg"),
        dpi=heatmap_dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        join(path_to_save, "figs", "pilr_2d.png"),
        dpi=heatmap_dpi,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.show()


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(description="Plot PIL distributions")
    parser.add_argument("path", type=Path, help="Data directory")
    parser.add_argument(
        "--x-min",
        type=float,
        default=Reader.PNM_MIN_RADIUS_NM,
        help="Minimum pore radius in nm (default: 0.003)",
    )
    parser.add_argument(
        "--r-points",
        type=int,
        default=900,
        help="Number of fitted radius grid points used for the heatmap",
    )
    parser.add_argument(
        "--r-step",
        type=int,
        default=1,
        help="Stride for radius grid points used for the heatmap",
    )
    parser.add_argument(
        "--l-bins",
        type=int,
        default=300,
        help="Number of length-grid nodes used for the heatmap",
    )
    parser.add_argument(
        "--heatmap-mode",
        choices=("smooth", "mesh"),
        default="smooth",
        help="Use a smoothed high-DPI image layer or a vector pcolormesh",
    )
    parser.add_argument(
        "--heatmap-interpolation",
        default="bicubic",
        help="Matplotlib interpolation used when --heatmap-mode=smooth",
    )
    parser.add_argument(
        "--heatmap-dpi",
        type=int,
        default=600,
        help="DPI used for SVG image layers and PNG output",
    )
    parser.add_argument(
        "--fit-sample-stride",
        type=int,
        default=10,
        help=(
            "Uniform subsample stride applied to radii before fitting P(r) "
            "(millions of edges otherwise); 1 uses every value"
        ),
    )
    args = parser.parse_args()

    plot_distributions(
        str(args.path),
        args.x_min,
        args.r_points,
        args.r_step,
        args.l_bins,
        args.heatmap_mode,
        args.heatmap_interpolation,
        args.heatmap_dpi,
        args.fit_sample_stride,
    )
