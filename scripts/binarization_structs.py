import argparse
import time
from pathlib import Path

import numpy as np

from processes.segmentation import BinarizeAlgo
from scripts.structure_image_utils import (
    build_segmentator,
    extract_settings,
    image_base_name,
    iter_structure_files,
    kprint,
    load_structure,
    write_binary_file,
)
from utils.logging_setup import setup_logging


def binarize_structures(
    structures_dir: Path,
    output_bin_dir: Path,
    output_raw_dir: Path,
    ref_size: int,
    dev: float,
    num_workers: int,
    atom_chunk: int,
) -> None:
    output_bin_dir.mkdir(parents=True, exist_ok=True)
    output_raw_dir.mkdir(parents=True, exist_ok=True)

    structure_files = iter_structure_files(structures_dir)
    for i, structure_file in enumerate(structure_files):
        structure = load_structure(structure_file)
        num, time_ps, bbox, resolution, img_size = extract_settings(
            structure, ref_size, dev
        )
        base_name = image_base_name(num, time_ps, bbox, resolution)
        binarized_path = output_bin_dir / f"{base_name}.npy"
        raw_path = output_raw_dir / f"{base_name}.raw"

        if binarized_path.exists() and raw_path.exists():
            kprint(
                f"Skip binarization with num={num}. Its {i + 1} from {len(structure_files)}"
            )
            continue

        segmentator = build_segmentator(structure, bbox, img_size)

        kprint(f"Run binarization for num={num}")
        start_time = time.time()
        img = 1 - segmentator.binarize(
            num_workers=num_workers,
            atom_chunk=atom_chunk,
            algo=BinarizeAlgo.PROCESS_CHUNK,
        )
        np.save(binarized_path, img)
        write_binary_file(img, raw_path)
        kprint(
            f"Binarization struct {num} is finished! Elapsed time: {time.time() - start_time}s. Its {i + 1} from {len(structure_files)}"
        )


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Binarize extracted structure .npz files."
    )
    parser.add_argument(
        "structures_dir", type=Path, help="Input structures dir"
    )
    parser.add_argument(
        "output_bin_dir", type=Path, help="Output bin image directory"
    )
    parser.add_argument(
        "output_raw_dir", type=Path, help="Output raw image directory"
    )
    parser.add_argument("--ref-size", type=int, required=True)
    parser.add_argument("--dev", type=float, default=2.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--atom-chunk",
        type=int,
        default=1024,
        help=(
            "Atoms per pairwise-distance batch inside each worker. Peak "
            "memory per concurrent slice is roughly "
            "ref_size^2 * atom_chunk * 9 bytes, so lowering this trades "
            "speed for memory instead of lowering --num-workers (see "
            "docs/reproduction.md sec. 15)."
        ),
    )
    args = parser.parse_args()

    binarize_structures(
        structures_dir=args.structures_dir,
        output_bin_dir=args.output_bin_dir,
        output_raw_dir=args.output_raw_dir,
        ref_size=args.ref_size,
        dev=args.dev,
        num_workers=args.num_workers,
        atom_chunk=args.atom_chunk,
    )


if __name__ == "__main__":
    main()
