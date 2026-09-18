import argparse
from pathlib import Path

import numpy as np
import numpy.lib.npyio as npyio
import pytest

from base.kerogendata import AtomData
from scripts.dynamic_struct_extractor import build_indexes_from_args
from scripts.structure_image_utils import (
    collect_indexes,
    iter_structure_files,
    load_structure,
    parse_indexes,
    save_structure,
    select_indexes_from_available,
)


def _make_atoms(count: int) -> list[AtomData]:
    return [
        AtomData(
            struct_number=i,
            struct_type="KRG",
            atom_id=f"C{i}",
            type_id=i % 5,
            pos=np.array([i * 0.1, i * 0.2, i * 0.3], dtype=np.float32),
        )
        for i in range(count)
    ]


def _write_empty_gro(path: Path, steps: list[int]) -> None:
    frames = [
        f"Kerogen t= {position:.5f} step= {step}\n"
        "0\n"
        "1.00000 1.00000 1.00000\n"
        for position, step in enumerate(steps)
    ]
    path.write_text("".join(frames), encoding="utf-8")


def test_load_structure_round_trips_save_structure(tmp_path: Path) -> None:
    atoms = _make_atoms(7)
    path = tmp_path / "struct.npz"
    save_structure(path, (42, 12.5, np.array(atoms), (1.0, 2.0, 3.0)))

    num, time_ps, loaded_atoms, size = load_structure(path)

    assert (num, time_ps, size) == (42, 12.5, (1.0, 2.0, 3.0))
    assert len(loaded_atoms) == 7
    assert loaded_atoms[3].struct_number == 3
    assert loaded_atoms[3].atom_id == "C3"
    np.testing.assert_allclose(loaded_atoms[3].pos, [0.3, 0.6, 0.9], atol=1e-6)


def test_load_structure_reads_each_array_once_not_once_per_atom(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression test: NpzFile.__getitem__ re-decompresses the whole array
    on every call (no caching) - indexing data[key][i] inside a per-atom
    loop is O(atom_count^2) and was observed to exhaust memory on a real
    ~66k-atom structure. Guard against that pattern coming back."""
    atom_count = 50
    path = tmp_path / "struct.npz"
    save_structure(
        path, (1, 0.0, np.array(_make_atoms(atom_count)), (1.0, 1.0, 1.0))
    )

    call_count = 0
    original_getitem = npyio.NpzFile.__getitem__

    def counting_getitem(self: npyio.NpzFile, key: str) -> object:
        nonlocal call_count
        call_count += 1
        return original_getitem(self, key)

    monkeypatch.setattr(npyio.NpzFile, "__getitem__", counting_getitem)

    load_structure(path)

    # 7 top-level keys (num, time_ps, size, struct_numbers, struct_types,
    # atom_ids, type_ids, positions), each read once - not scaled by
    # atom_count (which would be atom_count * 5 = 250 for this fixture).
    assert call_count <= 10, (
        f"load_structure() accessed the npz archive {call_count} times "
        f"for {atom_count} atoms - looks like it's re-reading an array "
        "per atom again instead of once"
    )


def test_parse_indexes_splits_commas_and_whitespace() -> None:
    assert parse_indexes(["25000", "50000,75000", "100000 125000"]) == [
        25000,
        50000,
        75000,
        100000,
        125000,
    ]


def test_collect_indexes_parses_cli_strings_to_ints(tmp_path: Path) -> None:
    indexes_file = tmp_path / "indexes.txt"
    indexes_file.write_text("300000,325000\n350000\n")

    result = collect_indexes(["25000", "50000,75000"], indexes_file)

    assert result == [25000, 50000, 75000, 300000, 325000, 350000]
    assert all(isinstance(index, int) for index in result)


def test_collect_indexes_deduplicates_and_requires_at_least_one() -> None:
    assert collect_indexes(["1", "1", "2"], None) == [1, 2]
    with pytest.raises(ValueError, match="Pass at least one"):
        collect_indexes([], None)


def test_select_indexes_from_available_returns_exact_count_with_endpoints() -> (
    None
):
    # Reproduces the P1-05 report: 6,560-step trajectory, 500 requested
    # structures must not silently become 501.
    available = list(range(25000, 25000 + 250000 * 6561, 250000))
    indexes = select_indexes_from_available(
        available,
        count=500,
        mode="all",
    )

    assert len(indexes) == 500
    assert len(set(indexes)) == 500
    assert indexes[0] == available[0]
    assert indexes[-1] == available[-1]


def test_select_indexes_from_available_handles_fewer_positions_than_requested() -> (
    None
):
    # Only 11 distinct integer positions exist between 0 and 10 inclusive.
    indexes = select_indexes_from_available(
        range(11),
        count=500,
        mode="all",
    )

    assert len(indexes) == 11
    assert indexes[0] == 0
    assert indexes[-1] == 10


def test_select_indexes_from_available_single_structure_is_the_start() -> None:
    indexes = select_indexes_from_available(
        [25000, 275000, 525000],
        count=1,
        mode="all",
    )
    assert indexes == [25000]


def test_select_indexes_from_available_part_returns_first_indexes() -> None:
    available = [0, 3, 10, 50, 51, 80, 200, 400, 900, 901, 2000]
    indexes = select_indexes_from_available(
        available,
        count=10,
        mode="part",
    )
    assert indexes == available[:10]


def test_select_indexes_from_available_supports_irregular_steps() -> None:
    indexes = select_indexes_from_available(
        [30000, 90000, 100000, 400000, 900000],
        count=3,
        mode="all",
    )

    assert indexes == [30000, 100000, 900000]


def test_dynamic_extractor_selects_500_frames_after_skipping_first(
    tmp_path: Path,
) -> None:
    trajectory = tmp_path / "trajectory.gro"
    steps = [25000 + 250000 * position for position in range(1001)]
    _write_empty_gro(trajectory, steps)
    args = argparse.Namespace(
        auto_indexes=True,
        input=trajectory,
        count_structures=500,
        mode="all",
        index=[],
        indexes_file=None,
    )

    indexes = build_indexes_from_args(args)

    assert len(indexes) == 500
    assert len(set(indexes)) == 500
    assert indexes[0] == steps[1]
    assert indexes[-1] == steps[-1]
    assert set(indexes) <= set(steps[1:])
    assert 25000 not in indexes


def test_dynamic_extractor_excludes_first_frame_from_explicit_indexes(
    tmp_path: Path,
) -> None:
    trajectory = tmp_path / "trajectory.gro"
    _write_empty_gro(trajectory, [25000, 275000, 525000])
    args = argparse.Namespace(
        auto_indexes=False,
        input=trajectory,
        count_structures=None,
        mode="all",
        index=["25000,275000"],
        indexes_file=None,
    )

    assert build_indexes_from_args(args) == [275000]


def test_iter_structure_files_returns_every_structure_sorted_by_step(
    tmp_path: Path,
) -> None:
    expected_names = [
        "struct-num=25_time-ps=0.1.npz",
        "struct-num=100_time-ps=0.4.npz",
        "struct-num=300_time-ps=1.2.npz",
    ]
    for name in reversed(expected_names):
        (tmp_path / name).touch()
    (tmp_path / "indexes.txt").touch()
    (tmp_path / "struct-num=50_time-ps=0.2.txt").touch()

    files = iter_structure_files(tmp_path)

    assert [path.name for path in files] == expected_names
