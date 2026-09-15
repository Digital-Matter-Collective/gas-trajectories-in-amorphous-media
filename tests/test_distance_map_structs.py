from pathlib import Path

import numpy as np
import pytest

import scripts.distance_map_structs as distance_map_structs


class _SegmentatorStub:
    def dist_map(self) -> np.ndarray:
        return np.array([[[0.0, 0.5]]], dtype=np.float32)


def test_build_distance_maps_writes_directly_to_output_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    structures_dir = tmp_path / "structures"
    structures_dir.mkdir()
    structure_path = structures_dir / "struct-num=7_time-ps=1.0.npz"
    structure_path.touch()
    output_float_dir = tmp_path / "distance_maps"

    structure = object()
    monkeypatch.setattr(
        distance_map_structs,
        "iter_structure_files",
        lambda path, indexes: [structure_path],
    )
    monkeypatch.setattr(
        distance_map_structs, "load_structure", lambda path: structure
    )
    monkeypatch.setattr(
        distance_map_structs,
        "extract_settings",
        lambda value, ref_size, dev: (7, 1.0, object(), 0.01, (1, 1, 2)),
    )
    monkeypatch.setattr(
        distance_map_structs, "image_base_name", lambda *args: "distance-map"
    )
    monkeypatch.setattr(
        distance_map_structs,
        "build_segmentator",
        lambda *args: _SegmentatorStub(),
    )

    distance_map_structs.build_distance_maps(
        structures_dir=structures_dir,
        output_float_dir=output_float_dir,
        ref_size=300,
        dev=4.0,
    )

    result_path = output_float_dir / "distance-map.npy"
    assert result_path.is_file()
    assert not (output_float_dir / "float_images").exists()
    np.testing.assert_array_equal(
        np.load(result_path), np.array([[[0.0, 0.5]]], dtype=np.float32)
    )


def test_main_passes_explicit_indexes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(distance_map_structs, "setup_logging", lambda: None)
    monkeypatch.setattr(
        distance_map_structs,
        "build_distance_maps",
        lambda **kwargs: captured.update(kwargs),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "distance_map_structs",
            str(tmp_path / "structures"),
            str(tmp_path / "float_images"),
            "--ref-size",
            "300",
            "--index",
            "25000,50000",
            "--index",
            "75000",
        ],
    )

    distance_map_structs.main()

    assert captured["indexes"] == [25000, 50000, 75000]
