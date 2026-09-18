import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Tuple

import numpy as np
import numpy.typing as npt

from base.boundingbox import BoundingBox, KerogenBox
from base.kerogendata import AtomData
from utils.utils import kprint  # noqa: F401 (re-exported for callers)

if TYPE_CHECKING:
    from processes.segmentation import Segmentator

# (num, time_ps, atoms, size) as extracted by base.reader.Reader and
# consumed/produced by save_structure()/load_structure().
Structure = Tuple[int, float, npt.NDArray[Any], Tuple[float, float, float]]

additional_radius = 0.0

atom_real_sizes = {
    i: s for i, s in enumerate([0.17, 0.152, 0.155, 0.109, 0.18])
}

ext_radius = {
    i: s
    for i, s in enumerate(
        [
            additional_radius,
            additional_radius,
            additional_radius,
            0.0,
            additional_radius,
        ]
    )
}

STRUCTURE_PATTERN = re.compile(
    r"struct-num=(?P<step>\d+)" r"_time-ps=(?P<time_ps>\d+(?:\.\d+)?)\.npz"
)

HEADER_PATTERN = re.compile(
    r"t=\s*(?P<time>[0-9]+(?:\.[0-9]+)?)\s+step=\s*(?P<step>[0-9]+)"
)


@dataclass(frozen=True)
class GroTrajectoryInfo:
    start_step: int
    step_size: int | None
    start_time: float
    time_step_size: float | None
    frame_count: int | None
    last_step: int | None
    available_steps: tuple[int, ...] | None


@dataclass(frozen=True)
class StepTimeMapping:
    anchor_step: int
    anchor_time_ps: float
    step_delta: int
    time_delta_ps: float

    def __post_init__(self) -> None:
        if self.step_delta == 0:
            raise ValueError("step_delta must be non-zero")

    def time_ps(self, step: int) -> float:
        return self.anchor_time_ps + (
            (step - self.anchor_step) * self.time_delta_ps / self.step_delta
        )


def get_size(type_id: int) -> float:
    return atom_real_sizes[type_id]


def get_ext_size(type_id: int) -> float:
    return ext_radius[type_id]


def write_binary_file(array: npt.NDArray[Any], file_name: Path) -> None:
    with file_name.open("wb") as file:
        for i in range(array.shape[2]):
            for j in range(array.shape[1]):
                file.write(bytes(bytearray(array[:, j, i])))


def parse_gro_header(line: str) -> tuple[int, float]:
    match = HEADER_PATTERN.search(line)
    if match is None:
        raise ValueError(f"Cannot parse trajectory header: {line.strip()}")
    time_ps = float(match.group("time"))
    step = int(match.group("step"))
    return step, time_ps


def scan_gro_trajectory_info(
    input_path: Path,
    count_all_frames: bool = True,
) -> GroTrajectoryInfo:
    steps: list[int] = []
    times: list[float] = []
    frame_count = 0

    with input_path.open("r", encoding="utf-8") as f:
        while True:
            header = f.readline()
            if not header:
                break

            step, time_ps = parse_gro_header(header)
            count_line = f.readline()
            if not count_line:
                raise RuntimeError(
                    f"Unexpected EOF after frame header: {header.strip()}"
                )
            try:
                atom_count = int(count_line.strip())
            except ValueError as exc:
                raise RuntimeError(
                    f"Cannot parse atom count after header: {header.strip()}"
                ) from exc

            for _ in range(atom_count + 1):
                if not f.readline():
                    raise RuntimeError(
                        f"Unexpected EOF while skipping frame {frame_count + 1}"
                    )

            frame_count += 1
            if count_all_frames or len(steps) < 2:
                steps.append(step)
                times.append(time_ps)

            if not count_all_frames and len(steps) == 2:
                break

    if not steps:
        raise RuntimeError(f"No frames found in {input_path}")

    step_size = steps[1] - steps[0] if len(steps) > 1 else None
    time_step_size = times[1] - times[0] if len(times) > 1 else None
    counted_frames = frame_count if count_all_frames else None
    last_step = steps[-1] if count_all_frames else None
    return GroTrajectoryInfo(
        start_step=steps[0],
        step_size=step_size,
        start_time=times[0],
        time_step_size=time_step_size,
        frame_count=counted_frames,
        last_step=last_step,
        available_steps=tuple(steps) if count_all_frames else None,
    )


def infer_step_time_mapping(input_path: Path) -> StepTimeMapping:
    """Infer an affine simulation-step to time mapping from two GRO frames."""
    info = scan_gro_trajectory_info(input_path, count_all_frames=False)
    if info.step_size is None or info.time_step_size is None:
        raise ValueError(
            f"Need at least two frames to infer step/time mapping: {input_path}"
        )
    return StepTimeMapping(
        anchor_step=info.start_step,
        anchor_time_ps=info.start_time,
        step_delta=info.step_size,
        time_delta_ps=info.time_step_size,
    )


def resolve_step_time_mapping(
    trajectory_path: Path | None,
    *,
    anchor_step: int | None = None,
    anchor_time_ps: float | None = None,
    step_delta: int | None = None,
    time_delta_ps: float | None = None,
) -> StepTimeMapping:
    """Resolve a mapping from a trajectory, with optional explicit overrides."""
    inferred = None
    explicit_values = (
        anchor_step,
        anchor_time_ps,
        step_delta,
        time_delta_ps,
    )
    if (
        any(value is None for value in explicit_values)
        and trajectory_path is not None
        and trajectory_path.is_file()
    ):
        inferred = infer_step_time_mapping(trajectory_path)

    values = {
        "anchor_step": anchor_step,
        "anchor_time_ps": anchor_time_ps,
        "step_delta": step_delta,
        "time_delta_ps": time_delta_ps,
    }
    if inferred is not None:
        for name in values:
            if values[name] is None:
                values[name] = getattr(inferred, name)

    missing = [name for name, value in values.items() if value is None]
    if missing:
        source = str(trajectory_path) if trajectory_path is not None else "none"
        raise ValueError(
            "Cannot resolve step/time mapping; missing "
            f"{', '.join(missing)} and trajectory source is {source}"
        )

    resolved_anchor_step = values["anchor_step"]
    resolved_anchor_time_ps = values["anchor_time_ps"]
    resolved_step_delta = values["step_delta"]
    resolved_time_delta_ps = values["time_delta_ps"]
    assert resolved_anchor_step is not None
    assert resolved_anchor_time_ps is not None
    assert resolved_step_delta is not None
    assert resolved_time_delta_ps is not None

    return StepTimeMapping(
        anchor_step=int(resolved_anchor_step),
        anchor_time_ps=float(resolved_anchor_time_ps),
        step_delta=int(resolved_step_delta),
        time_delta_ps=float(resolved_time_delta_ps),
    )


def select_indexes_from_available(
    available_indexes: Iterable[int],
    count: int,
    mode: str,
) -> list[int]:
    """Select at most ``count`` indexes from an ordered available set.

    ``mode="all"`` spreads selections by position, including both ends.
    ``mode="part"`` returns the first available indexes. Values need not be
    regularly spaced.
    """
    if count <= 0:
        raise ValueError("selection count must be positive")

    indexes = list(dict.fromkeys(available_indexes))
    if not indexes:
        raise ValueError("No indexes are available for selection")

    if mode == "all":
        if count >= len(indexes):
            return indexes
        if count == 1:
            positions = [0]
        else:
            last_position = len(indexes) - 1
            positions = sorted(
                {round(i * last_position / (count - 1)) for i in range(count)}
            )
        return [indexes[position] for position in positions]
    elif mode == "part":
        return indexes[:count]
    else:
        raise ValueError(f"Unknown mode: {mode}")


def parse_indexes(values: Iterable[str] | None) -> list[int]:
    indexes: list[int] = []
    for value in values or []:
        for item in value.replace(",", " ").split():
            indexes.append(int(item))
    return indexes


def read_indexes_file(path: Path) -> list[int]:
    return parse_indexes(path.read_text(encoding="utf-8").splitlines())


def collect_indexes(indexes: list[str], indexes_file: Path | None) -> list[int]:
    result = parse_indexes(indexes)
    if indexes_file is not None:
        result.extend(read_indexes_file(indexes_file))

    unique_result = list(dict.fromkeys(result))
    if not unique_result:
        raise ValueError("Pass at least one --index or --indexes-file")
    return unique_result


def structure_file_name(num: int, time_ps: int | float) -> str:
    return f"struct-num={num}_time-ps={time_ps}.npz"


def image_base_name(
    num: int,
    time_ps: int | float,
    bbox: BoundingBox,
    resolution: float,
) -> str:
    return (
        f"result-img-num={num}_time-ps={time_ps}"
        f"_bbox={bbox._short_str()}_resolution={resolution:.9f}"
    )


def iter_structure_files(
    structures_dir: Path,
) -> list[Path]:
    files_by_step = []

    for path in structures_dir.iterdir():
        if not path.is_file():
            continue
        match = STRUCTURE_PATTERN.fullmatch(path.name)
        if not match:
            continue
        step = int(match.group("step"))
        files_by_step.append((step, path))

    return [path for _, path in sorted(files_by_step, key=lambda item: item[0])]


def save_structure(path: Path, struct: Structure) -> None:
    num, time_ps, atoms, size = struct
    with path.open("wb") as f:
        np.savez(
            f,
            num=np.asarray(num),
            time_ps=np.asarray(time_ps),
            size=np.asarray(size, dtype=np.float64),
            struct_numbers=np.array(
                [a.struct_number for a in atoms], dtype=np.int32
            ),
            struct_types=np.array([a.struct_type for a in atoms], dtype="<U8"),
            atom_ids=np.array([a.atom_id for a in atoms], dtype="<U8"),
            type_ids=np.array([a.type_id for a in atoms], dtype=np.int8),
            positions=np.array([a.pos for a in atoms], dtype=np.float32),
        )


def load_structure(path: Path) -> Structure:
    with np.load(path) as data:
        num = int(data["num"])
        time_ps = float(data["time_ps"])
        size_arr = data["size"]
        size = (float(size_arr[0]), float(size_arr[1]), float(size_arr[2]))
        # NpzFile.__getitem__ re-reads and decompresses the whole array from
        # the zip archive on every call (no caching) - read each array once
        # here rather than once per atom, or this is O(atom_count^2).
        struct_numbers = data["struct_numbers"]
        struct_types = data["struct_types"]
        atom_ids = data["atom_ids"]
        type_ids = data["type_ids"]
        positions = data["positions"]
        atoms = np.array(
            [
                AtomData(
                    int(struct_numbers[i]),
                    str(struct_types[i]),
                    str(atom_ids[i]),
                    int(type_ids[i]),
                    positions[i],
                )
                for i in range(len(struct_numbers))
            ]
        )
    return num, time_ps, atoms, size


def extract_settings(
    structure: Structure, ref_size: int, dev: float
) -> Tuple[int, float, KerogenBox, float, Tuple[int, int, int]]:
    from processes.segmentation import Segmentator

    num, time_ps, atoms, size = structure
    bbox = Segmentator.cut_cell(size, dev)
    resolution = min(bbox.size()) / ref_size
    img_size = Segmentator.calc_image_size(
        bbox.size(), reference_size=ref_size, by_min=True
    )
    return num, time_ps, bbox, resolution, img_size


def build_segmentator(
    structure: Structure,
    bbox: BoundingBox,
    img_size: Tuple[int, int, int],
) -> "Segmentator":
    from base.kerogendata import KerogenData
    from base.periodizer import Periodizer
    from processes.segmentation import Segmentator

    _, _, atoms, _ = structure
    kerogen_data = KerogenData(None, list(atoms), bbox)
    if not kerogen_data.checkPeriodization():
        Periodizer.periodize(kerogen_data)

    return Segmentator(
        kerogen_data,
        img_size,
        size_data=get_size,
        radius_extention=get_ext_size,
        partitioning=2,
    )
