import argparse
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import repeat
from pathlib import Path
from typing import Callable, List, Tuple, cast

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from base.trajectory import Trajectory
from utils.cache_manifest import check_cache, write_manifest
from utils.logging_setup import setup_logging
from utils.utils import kprint

_IMAGE_PATTERN = re.compile(
    r"result-img-num=\d+_time-ps=(?P<time_ps>\d+(?:\.\d+)?).*\.npy$"
)


@dataclass
class RMSDResult:
    rmsd: np.ndarray
    t: np.ndarray


@dataclass(frozen=True)
class PackedImages:
    words: npt.NDArray[np.uint64]
    sizes: npt.NDArray[np.int64]
    image_shape: tuple[int, ...]


_worker_buffers = threading.local()


def parse_trj(value: str) -> tuple[Path, str]:
    """Parse 'path/to/trj.gro:LABEL' into (Path, label)."""
    parts = value.rsplit(":", 1)
    if len(parts) != 2 or not parts[1]:
        raise argparse.ArgumentTypeError(f"Expected path:label, got '{value}'")
    return Path(parts[0]), parts[1]


def scan_image_infos(images_dir: Path) -> list[tuple[float, str]]:
    """Scan directory for binarized .npy image files, extract time stamps from names."""
    infos = []
    for path in images_dir.iterdir():
        match = _IMAGE_PATTERN.match(path.name)
        if match is None:
            continue
        time_ps = float(match.group("time_ps"))
        infos.append((time_ps, str(path)))
    if not infos:
        raise RuntimeError(f"No matching image files found in {images_dir}")
    return sorted(infos, key=lambda x: x[0])


def extract_mean_displacement(
    trajectories: List[Trajectory], traj_stride: int = 1
) -> RMSDResult:
    a_msd = []
    a_t = []

    for _, trj in enumerate(trajectories[::traj_stride]):
        msd = trj.msd_average_time()
        t = np.asarray(trj.times, dtype=float)
        a_t.append((t * 1e-6).astype(float))
        a_msd.append(msd)

    msd_mean = np.mean(np.stack(a_msd, axis=0), axis=0)
    rmsd = np.sqrt(msd_mean)
    return RMSDResult(rmsd, a_t[0])


def plot_corrfunc_and_md(
    dt: npt.NDArray[np.float64],
    C_t: npt.NDArray[np.float64],
    trj_msd_list: List[tuple[RMSDResult, str]],
    save_path: Path | None = None,
    max_t: float = 2.8,
    x_max: float | None = None,
) -> None:
    dt = np.asarray(dt, dtype=float)
    C_t = np.asarray(C_t, dtype=float)
    mask = dt <= max_t
    mask[0] = False
    dt = dt[mask]
    C_t = C_t[mask]

    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_ct = default_colors[0]
    color_md = default_colors[1]

    fig, ax1 = plt.subplots(figsize=(9, 5))

    line1 = ax1.plot(
        dt,
        C_t,
        linewidth=3.0,
        color=color_ct,
        label=r"$C(t)$",
    )
    ax1.set_xlabel(r"Time delay, $\mu$s", fontsize=20)

    ax1.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax1.tick_params(axis="both", labelsize=16)
    ax1.tick_params(axis="y", labelcolor=color_ct)

    lines = line1
    styles = ['-', ':', '--', '-.', '.']
    ax2 = ax1.twinx()
    for res, prefix in trj_msd_list:
        label = prefix + " " + r"$RMSD(t)$"

        r = res.rmsd
        t = res.t

        mask = t <= max_t
        mask[0] = False
        t = t[mask]
        r = r[mask]

        line = ax2.plot(
            t,
            r,
            linewidth=3.0,
            color=color_md,
            label=label,
            linestyle=styles.pop(0),
        )
        lines += line

    ax2.set_yscale("log")
    ax2.set_xscale("log")
    ax1.set_xscale("log")
    if x_max is not None:
        ax1.set_xlim(right=x_max)
        ax2.set_xlim(right=x_max)
    ax2.set_ylabel(r"$\mathrm{RMSD}(t)$, nm", fontsize=20, color=color_md)
    ax2.tick_params(axis="both", labelsize=16)
    ax2.tick_params(axis="y", labelcolor=color_md)

    positive_r = []
    for res, _ in trj_msd_list:
        r = np.asarray(res.rmsd, dtype=float)
        positive_r.extend(r[np.isfinite(r) & (r > 0)])

    if positive_r:
        ax2.set_ylim(bottom=0.09)
        ax2.set_ylim(top=max(positive_r) * 10)

    labels = [str(line.get_label()) for line in lines]
    ax1.legend(
        lines,
        labels,
        # loc="upper left",
        bbox_to_anchor=(0.4, 0.8),
        borderaxespad=0.0,
        fontsize=18,
        frameon=False,
    )

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def correlation_average_time(
    image_infos: List[tuple[float, str]],
    load_img: Callable[[str], npt.NDArray[np.int8]],
    ct_save_path: Path,
    num_workers: int = 4,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Time-averaged autocorrelation function:

        C(tau_k) = mean_i [ sum_x I_i(x) I_{i+k}(x) / sum_x I_i(x) ]
    """
    image_infos = sorted(image_infos, key=lambda x: x[0])

    times_ps = np.array([info[0] for info in image_infos], dtype=np.float64)
    img_files = [info[1] for info in image_infos]

    n = len(image_infos)
    dt = (times_ps - times_ps[0]) * 1e-6  # ps → μs
    kprint(f"dt range: {dt[1]:.6f} … {dt[-1]:.3f} μs  ({n} frames)")

    cache_metadata = {
        "frame_count": n,
        "image_file_names": sorted(str(f) for f in img_files),
    }
    status = check_cache(ct_save_path, cache_metadata)
    if status == "match":
        C_t = np.load(ct_save_path)
    elif status == "legacy":
        kprint(
            f"Upgrading legacy cache {ct_save_path} to provenance-tracked "
            "format (trusted as-is, not recomputed)"
        )
        C_t = np.load(ct_save_path)
        write_manifest(ct_save_path, cache_metadata)
    else:
        if status == "mismatch":
            kprint(
                f"Cache {ct_save_path} does not match current images; "
                "recomputing from scratch"
            )
        C_t = np.full(n, np.nan, dtype=np.float64)
        write_manifest(ct_save_path, cache_metadata)

    if C_t.shape != (n,):
        kprint(
            f"Cache {ct_save_path} has shape {C_t.shape}, expected {(n,)}; "
            "recomputing from scratch"
        )
        C_t = np.full(n, np.nan, dtype=np.float64)

    missing_lags = np.flatnonzero(np.isnan(C_t))
    if len(missing_lags) == 0:
        return dt, C_t

    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")

    load_start = time.perf_counter()
    kprint("Loading and bit-packing binary images (one pass over each file)")
    packed = pack_binary_images(img_files, load_img)
    packed_mib = packed.words.nbytes / (1024**2)
    kprint(
        f"Loaded and bit-packed {n} images with shape {packed.image_shape} "
        f"into {packed_mib:.1f} MiB in "
        f"{time.perf_counter() - load_start:.2f} sec"
    )

    executor = (
        ThreadPoolExecutor(max_workers=num_workers) if num_workers > 0 else None
    )
    try:
        for lag in range(n):
            start_time = time.perf_counter()
            if not np.isnan(C_t[lag]):
                kprint(f"Skip lag: {lag} from {n}, C: {C_t[lag]}")
                continue

            pair_count = n - lag
            if executor is None:
                values = _process_packed_pairs(range(pair_count), lag, packed)
            else:
                chunk_size = (pair_count + num_workers - 1) // num_workers
                chunks = [
                    range(start, min(start + chunk_size, pair_count))
                    for start in range(0, pair_count, chunk_size)
                ]
                chunk_results = list(
                    executor.map(
                        _process_packed_pairs,
                        chunks,
                        repeat(lag),
                        repeat(packed),
                    )
                )
                values = np.concatenate(chunk_results)

            C_t[lag] = np.mean(values)
            np.save(ct_save_path, C_t)
            kprint(
                f"Ready lag: {lag} from {n}, C: {C_t[lag]} in "
                f"{time.perf_counter() - start_time:.2f} sec"
            )
    finally:
        if executor is not None:
            executor.shutdown()

    return dt, C_t


def pack_binary_images(
    img_files: List[str],
    load_img: Callable[[str], npt.NDArray[np.int8]],
) -> PackedImages:
    """Load binary images once and pack every 64 voxels into one word."""
    if not img_files:
        raise ValueError("At least one image is required")

    first_image = np.asarray(load_img(img_files[0]))
    image_shape = first_image.shape
    voxel_count = first_image.size
    word_count = (voxel_count + 63) // 64

    words = np.zeros((len(img_files), word_count), dtype=np.uint64)
    byte_rows = words.view(np.uint8).reshape(len(img_files), word_count * 8)
    sizes = np.empty(len(img_files), dtype=np.int64)

    def store_image(index: int, image: npt.NDArray[np.int8]) -> None:
        image_array = np.asarray(image)
        if image_array.shape != image_shape:
            raise ValueError(
                f"Image {img_files[index]} has shape {image_array.shape}, "
                f"expected {image_shape}"
            )

        flat_image = image_array.reshape(-1)
        image_size = int(np.sum(flat_image, dtype=np.int64))
        if image_size <= 0:
            raise ValueError(f"Image {img_files[index]} contains no set voxels")

        packed_bytes = np.packbits(flat_image, bitorder="little")
        byte_rows[index, : len(packed_bytes)] = packed_bytes
        sizes[index] = image_size

    store_image(0, first_image)
    del first_image
    progress_step = max(1, len(img_files) // 10)
    for index, file_name in enumerate(img_files[1:], start=1):
        store_image(index, load_img(file_name))
        completed = index + 1
        if len(img_files) >= 20 and (
            completed % progress_step == 0 or completed == len(img_files)
        ):
            kprint(f"Bit-packed {completed} from {len(img_files)} images")

    return PackedImages(words=words, sizes=sizes, image_shape=image_shape)


def _process_packed_pairs(
    indexes: range,
    lag: int,
    packed: PackedImages,
) -> npt.NDArray[np.float64]:
    """Compute normalized intersections for a range of frame pairs."""
    word_count = packed.words.shape[1]
    and_words = getattr(_worker_buffers, "and_words", None)
    bit_counts = getattr(_worker_buffers, "bit_counts", None)
    if (
        and_words is None
        or bit_counts is None
        or and_words.shape != (word_count,)
    ):
        and_words = np.empty(word_count, dtype=np.uint64)
        bit_counts = np.empty(word_count, dtype=np.uint8)
        _worker_buffers.and_words = and_words
        _worker_buffers.bit_counts = bit_counts

    result = np.empty(len(indexes), dtype=np.float64)
    for local_index, image_index in enumerate(indexes):
        np.bitwise_and(
            packed.words[image_index],
            packed.words[image_index + lag],
            out=and_words,
        )
        np.bitwise_count(and_words, out=bit_counts)
        overlap = np.sum(bit_counts, dtype=np.uint64)
        result[local_index] = overlap / packed.sizes[image_index]
    return result


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(
        description=(
            "Compute time-averaged autocorrelation C(t) from binarized images "
            "and plot together with RMSD(t) trajectories."
        )
    )
    parser.add_argument(
        "images_dir",
        type=Path,
        help="Directory containing binarized .npy images (output of binarization_structs).",
    )
    parser.add_argument(
        "ct_file",
        type=Path,
        help="Path to save/load the C(t) cache (.npy). Computed incrementally.",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output figure path (e.g. figs/corrfunc.svg).",
    )
    parser.add_argument(
        "--trj",
        action="append",
        type=parse_trj,
        required=True,
        metavar="PATH:LABEL",
        help=(
            "Trajectory .gro file with a display label, e.g. trj.gro:CH4. "
            "Can be repeated for multiple gases."
        ),
    )
    parser.add_argument(
        "--max-t",
        type=float,
        default=2.8,
        metavar="US",
        help="Maximum time in μs shown on the plot (default: 2.8).",
    )
    parser.add_argument(
        "--x-max",
        type=float,
        default=None,
        metavar="US",
        help="Right X-axis display limit in μs (default: auto from data).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help=(
            "Number of threads for bit-packed C(t) computation; memory "
            "bandwidth usually saturates around 4 threads (default: 4)."
        ),
    )

    args = parser.parse_args()

    images_dir: Path = args.images_dir
    ct_file: Path = args.ct_file
    output: Path = args.output

    image_infos = scan_image_infos(images_dir)
    kprint(f"Found {len(image_infos)} image files in {images_dir}")

    def load_img(file_name: str) -> np.ndarray:
        img = np.load(file_name, mmap_mode="r")
        return cast(np.ndarray, np.equal(img, 0))

    ct_file.parent.mkdir(parents=True, exist_ok=True)
    dt, C_t = correlation_average_time(
        image_infos,
        load_img,
        ct_file,
        args.num_workers,
    )
    kprint(f"C(t) time range: {dt[0]:.4f} … {dt[-1]:.4f} μs")

    trj_msd_list = []
    for trj_path, label in args.trj:
        trajectories = Trajectory.read_trajectories(trj_path)
        res = extract_mean_displacement(trajectories)
        trj_msd_list.append((res, label))

    output.parent.mkdir(parents=True, exist_ok=True)
    plot_corrfunc_and_md(
        dt=dt,
        C_t=C_t,
        trj_msd_list=trj_msd_list,
        save_path=output,
        max_t=args.max_t,
        x_max=args.x_max,
    )


if __name__ == "__main__":
    main()
