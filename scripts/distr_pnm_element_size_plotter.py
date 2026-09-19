import argparse
from os import listdir
from os.path import isfile, join
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.signal import savgol_filter

from base.reader import Reader
from scripts.structure_image_utils import (
    StepTimeMapping,
    resolve_step_time_mapping,
)
from utils.logging_setup import setup_logging
from utils.utils import kprint


def compute_density(
    data: np.ndarray,
    bins: int = 50,
    smooth: bool = True,
    window_length: int = 25,
    polyorder: int = 3,
    x_min: float = Reader.PNM_MIN_RADIUS_NM,
    x_max: float = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Строит нормированную гистограмму-плотность.
    Возвращает:
        x  - центры бинов
        pn - значения плотности
    """
    p, bb = np.histogram(data, bins=bins, density=False)
    xdel = bb[1] - bb[0]
    x = bb[:-1] + 0.5 * xdel
    pn = p / np.sum(p * xdel)

    if smooth and len(pn) >= 7:
        # window_length должен быть нечётным и <= len(pn)
        wl = min(window_length, len(pn) if len(pn) % 2 == 1 else len(pn) - 1)
        if wl >= polyorder + 2 and wl % 2 == 1:
            pn = savgol_filter(pn, wl, polyorder)
            pn[pn < 0] = 0.0

    # Author decision 2026-09-02: the trailing 10 bins are a
    # not-useful-for-visualization tail; dropping them without renormalizing
    # is intentional here, not a bug.
    x = x[:-10]
    pn = pn[:-10]
    mask = np.logical_and(x > x_min, x < x_max)
    x = x[mask]
    pn = pn[mask]

    return x, pn


def plot_fill(
    x: np.ndarray, p: np.ndarray, time_us: float, color: str
) -> Poly3DCollection:
    verts_l = [(x, time_us, z) for x, z in zip(x, p)]
    verts_l += [(x[-1], time_us, 0.0), (x[0], time_us, 0.0)]

    return Poly3DCollection(
        [verts_l],
        facecolors=color,
        alpha=0.12,
        edgecolors="none",
    )


def get_sorted_pnm_files(path_to_pnms: str) -> list[tuple[int, str]]:
    onlyfiles = [
        f for f in listdir(path_to_pnms) if isfile(join(path_to_pnms, f))
    ]
    onlyfiles = [
        file
        for file in onlyfiles
        if "_link1" in file and file.startswith("pnm-num")
    ]

    steps = [int((file.split("=")[1]).split("_")[0]) for file in onlyfiles]
    return sorted(zip(steps, onlyfiles), key=lambda x: x[0])


def convert_step_to_time_us(step: int, mapping: StepTimeMapping) -> float:
    return mapping.time_ps(step) * 1e-6


def build_3d_distributions(
    paths: Tuple[str, str, str],
    step_time_mapping: StepTimeMapping,
    pnm_step: int = 40,
    bins: int = 50,
    smooth: bool = True,
    border: float = Reader.PNM_MIN_RADIUS_NM,
    x_min: float = Reader.PNM_MIN_RADIUS_NM,
    x_max: float = 2,
    scale: float = Reader.PNM_M_TO_NM,
) -> None:
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Фиксированные цвета по типу распределения
    color_throat = "tab:blue"  # P(l)
    color_pore = "tab:orange"  # P(r)

    is_fill = True

    label_added_pl = False
    label_added_pr = False

    path_to_pnms, dataset_label, path_to_save = paths
    kprint(path_to_pnms)
    sorted_files = get_sorted_pnm_files(path_to_pnms)

    # Author decision 2026-09-02: skipping the earliest frame(s) here is
    # intentional (pre-equilibration exclusion), same rationale as
    # `stationarity.py` excluding its first frame — not a bug.
    for step, file in sorted_files[pnm_step::pnm_step]:
        base_path = join(path_to_pnms, file[:-10])
        radiuses, throat_lengths = Reader.read_pnm_data(
            base_path,
            scale=scale,
            border=border,
        )

        time_us = convert_step_to_time_us(step, step_time_mapping)
        kprint(f"{dataset_label}: step={step}, time={time_us:.2f} us")

        # P(l)
        x_l, p_l = compute_density(
            throat_lengths,
            bins=bins,
            smooth=smooth,
            x_min=x_min,
            x_max=x_max,
        )
        y_l = np.full_like(x_l, time_us, dtype=float)
        ax.plot(
            x_l,
            y_l,
            p_l,
            color=color_throat,
            label=r"$P(h)$" if not label_added_pl else None,
            linewidth=2.5,
        )
        ax.text(
            x_l[-1] - 0.01,  # небольшой сдвиг вправо
            y_l[-1] - 0.5,  # то же время
            p_l[-1] + 0.5,  # конец кривой по z
            rf"{time_us:.2f}",
            zdir=(1, -0.5, 0),  # размещаем текст в плоскости x-z
            color=color_throat,
            fontsize=12,
            ha="left",
            va="center",
        )
        label_added_pl = True

        # P(r)
        x_r, p_r = compute_density(
            radiuses,
            bins=bins,
            smooth=smooth,
            x_min=x_min,
            x_max=x_max,
        )
        p_r = p_r * 0.5
        y_r = np.full_like(x_r, time_us, dtype=float)
        ax.plot(
            x_r,
            y_r,
            p_r,
            color=color_pore,
            label=r"$P(r)$" if not label_added_pr else None,
            linewidth=2.5,
        )
        label_added_pr = True

        if is_fill:
            collection_pl_fill = plot_fill(x_l, p_l, time_us, color_throat)
            collection_pr_fill = plot_fill(x_r, p_r, time_us, color_pore)
            ax.add_collection3d(collection_pl_fill)
            ax.add_collection3d(collection_pr_fill)
        else:
            ax.plot(
                x_l,
                y_l,
                np.zeros_like(p_l),
                color=color_throat,
                alpha=0.22,
                linewidth=1.2,
            )
            ax.plot(
                x_r,
                y_r,
                np.zeros_like(p_r),
                color=color_pore,
                alpha=0.22,
                linewidth=1.2,
            )

    if not is_fill:
        y_back = ax.get_ylim()[1]
        ax.plot(
            x_l,
            np.full_like(x_l, y_back),
            p_l,
            color=color_throat,
            alpha=0.15,
            linewidth=1.0,
        )

    ax.set_xlabel(r"$r, h$ (nm)", fontsize=20, labelpad=-10)
    ax.set_ylabel(r"Time ($\mu$s)", fontsize=20, labelpad=-10)
    ax.set_zlabel("Density", fontsize=20, labelpad=-6)

    xlabels = ax.get_xticklabels()
    res_xlabels = []
    kprint("xlabels=", list(enumerate(xlabels)))
    for i, x in enumerate(xlabels):
        if i in [4, 5, 6, 7]:
            x._text = ""
        res_xlabels.append(x)

    ax.set_xticklabels(xlabels)
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.tick_params(axis="z", labelsize=11)

    ax.view_init(elev=15, azim=-86)
    ax.legend(frameon=False, fontsize=12)

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # белые плоскости
    ax.xaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))

    # можно ещё убрать серые границы плоскостей
    ax.xaxis.pane.set_edgecolor("white")
    ax.yaxis.pane.set_edgecolor("white")
    ax.zaxis.pane.set_edgecolor("white")
    ax.grid(False)

    # ax.xaxis._axinfo["juggled"] = (1, 0, 2)
    # ax.yaxis._axinfo["juggled"] = (0, 1, 2)
    # ax.zaxis._axinfo["juggled"] = (0, 2, 1)

    ax.xaxis._axinfo["juggled"] = (1, 0, 2)
    ax.yaxis._axinfo["juggled"] = (0, 1, 2)
    ax.zaxis._axinfo["juggled"] = (1, 2, 0)
    ax.set_proj_type("ortho")

    for e, a in [(22, -72)]:
        ax.view_init(elev=e, azim=a)
        f_name = "fill" if is_fill else "proj"
        fig.savefig(
            join(path_to_save, f"{f_name}_elev_{e}_azim_{a}.svg"),
            # bbox_inches="tight",
        )

    # fig.savefig("all_distributions_3d.svg", bbox_inches="tight")
    plt.show()


if __name__ == '__main__':
    setup_logging()
    parser = argparse.ArgumentParser(
        description="3D PNM element size distributions"
    )
    parser.add_argument("pnm_dir", type=Path, help="PNM directory")
    parser.add_argument("figs_dir", type=Path, help="Output figures directory")
    parser.add_argument("--label", type=str, default="", help="Series label")
    parser.add_argument("--pnm-step", type=int, default=100)
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument("--no-smooth", action="store_true")
    parser.add_argument(
        "--x-min",
        type=float,
        default=Reader.PNM_MIN_RADIUS_NM,
        help="Strict lower display bound in nm (default: 0.003)",
    )
    parser.add_argument("--x-max", type=float, default=2.0)
    parser.add_argument(
        "--trajectory",
        type=Path,
        help=(
            "GRO trajectory used to infer step/time mapping "
            "(default: <pnm_dir>/../trj.gro)"
        ),
    )
    parser.add_argument("--anchor-step", type=int)
    parser.add_argument("--anchor-time-ps", type=float)
    parser.add_argument("--step-delta", type=int)
    parser.add_argument("--time-delta-ps", type=float)
    args = parser.parse_args()

    trajectory_path = args.trajectory or args.pnm_dir.parent / "trj.gro"
    step_time_mapping = resolve_step_time_mapping(
        trajectory_path,
        anchor_step=args.anchor_step,
        anchor_time_ps=args.anchor_time_ps,
        step_delta=args.step_delta,
        time_delta_ps=args.time_delta_ps,
    )
    kprint(f"Step/time mapping: {step_time_mapping}")
    build_3d_distributions(
        (str(args.pnm_dir), args.label, str(args.figs_dir)),
        step_time_mapping,
        pnm_step=args.pnm_step,
        bins=args.bins,
        smooth=not args.no_smooth,
        x_min=args.x_min,
        x_max=args.x_max,
    )
