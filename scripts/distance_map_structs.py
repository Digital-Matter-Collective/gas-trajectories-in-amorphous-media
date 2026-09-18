import argparse
import time
from pathlib import Path

import numpy as np

from scripts.structure_image_utils import (
    build_segmentator,
    extract_settings,
    image_base_name,
    iter_structure_files,
    kprint,
    load_structure,
)
from utils.logging_setup import setup_logging


def build_distance_maps(
    structures_dir: Path,
    output_dir: Path,
    ref_size: int,
    dev: float,
) -> None:
    float_image_dir = output_dir / "float_images"
    float_image_dir.mkdir(parents=True, exist_ok=True)

    structure_files = iter_structure_files(structures_dir)
    for i, structure_file in enumerate(structure_files):
        structure = load_structure(structure_file)
        num, time_ps, bbox, resolution, img_size = extract_settings(
            structure, ref_size=ref_size, dev=dev
        )
        segmentator = build_segmentator(structure, bbox, img_size)
        base_name = image_base_name(num, time_ps, bbox, resolution)
        float_path = float_image_dir / f"{base_name}.npy"

        if float_path.exists():
            kprint(
                f"Skip distance map with num={num}. Its {i + 1} from {len(structure_files)}"
            )
            continue

        kprint(f"Run distance map for num={num}")
        start_time = time.time()
        float_img = segmentator.dist_map()
        np.save(float_path, float_img)
        kprint(
            f"Distance map struct {num} is finished! Elapsed time: {time.time() - start_time}s. Its {i + 1} from {len(structure_files)}"
        )


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Build distance maps for extracted structure .npz files."
    )
    parser.add_argument(
        "structures_dir", type=Path, help="Input structures dir"
    )
    parser.add_argument("output_dir", type=Path, help="Output base directory")
    parser.add_argument("--ref-size", type=int, required=True)
    parser.add_argument("--dev", type=float, default=4.0)

    args = parser.parse_args()

    build_distance_maps(
        structures_dir=args.structures_dir,
        output_dir=args.output_dir,
        ref_size=args.ref_size,
        dev=args.dev,
    )


if __name__ == "__main__":
    main()
