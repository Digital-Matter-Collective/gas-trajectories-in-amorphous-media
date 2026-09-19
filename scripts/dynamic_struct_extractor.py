import argparse
import time
from pathlib import Path

import numpy as np

from scripts.structure_image_utils import (
    collect_indexes,
    kprint,
    save_structure,
    scan_gro_trajectory_info,
    select_indexes_from_available,
    structure_file_name,
)
from utils.logging_setup import setup_logging


def extract_structures(
    input_path: Path,
    output_dir: Path,
    indexes: list[int],
    slice_len: int = 100,
) -> None:
    from base.reader import Reader

    output_dir.mkdir(parents=True, exist_ok=True)

    aindexes = np.asarray(indexes, dtype=np.int32)
    existing_mask = np.asarray(
        [
            any(output_dir.glob(f"struct-num={num}_time-ps=*.npz"))
            for num in aindexes
        ],
        dtype=bool,
    )
    aindexes = aindexes[~existing_mask]

    count_steps = len(aindexes) // slice_len + 1
    for i in range(count_steps):
        start_time = time.time()
        start = i * slice_len
        stop = min((i + 1) * slice_len, len(aindexes))
        cur_indexes = aindexes[start:stop].tolist()
        if not cur_indexes:
            continue

        structures = Reader.read_structures_by_num(str(input_path), cur_indexes)
        for struct in structures:
            num, time_ps, _, _ = struct
            save_path = output_dir / structure_file_name(num, time_ps)
            save_structure(save_path, struct)

        kprint(f"Count structures step: {i + 1} from {count_steps}")
        kprint(f"Reading finished! Elapsed time: {time.time() - start_time}s")


def build_indexes_from_args(args: argparse.Namespace) -> list[int]:
    if args.auto_indexes:
        info = scan_gro_trajectory_info(args.input, count_all_frames=True)
        if info.available_steps is None or len(info.available_steps) < 2:
            raise ValueError("Need at least two trajectory frames")
        if len(set(info.available_steps)) != len(info.available_steps):
            raise ValueError("Trajectory frame step numbers must be unique")

        eligible_steps = (
            info.available_steps
            if args.include_first_step
            else info.available_steps[1:]
        )
        indexes = select_indexes_from_available(
            eligible_steps,
            count=args.count_structures,
            mode=args.mode,
        )
        kprint(
            "Trajectory info: "
            f"first_available_step={info.start_step}, "
            f"first_step_included={args.include_first_step}, "
            f"first_selected_step={indexes[0]}, "
            f"last_selected_step={indexes[-1]}, "
            f"frame_count={info.frame_count}"
        )
        kprint(f"Generated structure indexes: {len(indexes)}")
        return indexes

    indexes = collect_indexes(args.index, args.indexes_file)
    if args.include_first_step:
        return indexes
    first_step = scan_gro_trajectory_info(
        args.input, count_all_frames=False
    ).start_step
    filtered_indexes = [index for index in indexes if index != first_step]
    if len(filtered_indexes) != len(indexes):
        kprint(f"Skipping first trajectory frame with step={first_step}")
    if not filtered_indexes:
        raise ValueError(
            "The first trajectory frame cannot be extracted "
            "(pass --include-first-step to allow it)"
        )
    return filtered_indexes


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Extract selected structures from a .gro trajectory into .npz files."
    )
    parser.add_argument("input", type=Path, help="Input .gro trajectory file")
    parser.add_argument(
        "output_dir", type=Path, help="Directory for structures"
    )
    parser.add_argument(
        "--index",
        action="append",
        default=[],
        help=(
            "Structure step number. Can be repeated or comma-separated; "
            "the first trajectory frame is skipped unless "
            "--include-first-step is passed."
        ),
    )
    parser.add_argument(
        "--indexes-file",
        type=Path,
        help="Text file with structure step numbers separated by whitespace or commas.",
    )
    parser.add_argument(
        "--auto-indexes",
        action="store_true",
        help=(
            "Read available steps from trajectory headers, skip the first "
            "frame unless --include-first-step is passed, and select "
            "indexes by mode/count."
        ),
    )
    parser.add_argument(
        "--include-first-step",
        action="store_true",
        help=(
            "Extract the very first trajectory frame too. Off by default: "
            "it precedes equilibration and is not a valid stationarity "
            "baseline, so downstream PNM/structure-derived analyses "
            "(including gas-traj-stationarity) expect it to be absent."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["all", "part"],
        default="all",
        help=(
            "Selection mode for --auto-indexes: evenly spread or first part "
            "(default: all)."
        ),
    )
    parser.add_argument(
        "--count-structures",
        type=int,
        default=100,
        help="How many structures to request in --auto-indexes mode (default: 100).",
    )
    parser.add_argument(
        "--slice-len",
        type=int,
        default=100,
        help="How many requested structures to read per batch.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print generated indexes and exit without extracting structures.",
    )

    args = parser.parse_args()
    if args.auto_indexes and (args.index or args.indexes_file):
        parser.error("Use either --auto-indexes or --index/--indexes-file")

    indexes = build_indexes_from_args(args)
    if args.dry_run:
        preview = ", ".join(str(index) for index in indexes[:20])
        if len(indexes) > 20:
            preview += ", ..."
        kprint(f"Indexes preview: {preview}")
        return

    extract_structures(args.input, args.output_dir, indexes, args.slice_len)


if __name__ == "__main__":
    main()
