from pathlib import Path

import pytest

import scripts.binarization_structs as binarization_structs
import scripts.dynamic_struct_extractor as dynamic_struct_extractor
import scripts.stationarity as stationarity
from scripts.structure_image_utils import StepTimeMapping


def test_binarization_cli_uses_documented_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(binarization_structs, "setup_logging", lambda: None)
    monkeypatch.setattr(
        binarization_structs,
        "binarize_structures",
        lambda **kwargs: captured.update(kwargs),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "binarization_structs",
            str(tmp_path / "structures"),
            str(tmp_path / "bin_images"),
            str(tmp_path / "raw_images"),
        ],
    )

    binarization_structs.main()

    assert captured["ref_size"] == 600
    assert captured["dev"] == 4.0


def test_structure_extraction_cli_uses_documented_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def capture_args(args: object) -> list[int]:
        captured["count_structures"] = getattr(args, "count_structures")
        captured["mode"] = getattr(args, "mode")
        return [25000]

    monkeypatch.setattr(dynamic_struct_extractor, "setup_logging", lambda: None)
    monkeypatch.setattr(
        dynamic_struct_extractor, "build_indexes_from_args", capture_args
    )
    monkeypatch.setattr(
        dynamic_struct_extractor, "kprint", lambda message: None
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "dynamic_struct_extractor",
            str(tmp_path / "trajectory.gro"),
            str(tmp_path / "structures"),
            "--auto-indexes",
            "--dry-run",
        ],
    )

    dynamic_struct_extractor.main()

    assert captured == {"count_structures": 100, "mode": "all"}


@pytest.mark.parametrize(
    ("x_min_args", "expected_x_min"),
    [([], 0.015), (["--x-min", "0.02"], 0.02)],
)
def test_stationarity_cli_accepts_mapping_without_trajectory_and_x_min(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    x_min_args: list[str],
    expected_x_min: float,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(stationarity, "setup_logging", lambda: None)
    monkeypatch.setattr(stationarity, "kprint", lambda message: None)
    monkeypatch.setattr(
        stationarity,
        "analysis",
        lambda pnm_path, outdir, mapping, x_min: captured.update(
            pnm_path=pnm_path,
            outdir=outdir,
            mapping=mapping,
            x_min=x_min,
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "stationarity",
            str(tmp_path / "pnm"),
            str(tmp_path / "ks_stationarity"),
            "--anchor-step",
            "25000",
            "--anchor-time-ps",
            "50",
            "--step-delta",
            "250000",
            "--time-delta-ps",
            "500",
        ]
        + x_min_args,
    )

    stationarity.main()

    assert captured["mapping"] == StepTimeMapping(25000, 50.0, 250000, 500.0)
    assert captured["pnm_path"] == str(tmp_path / "pnm")
    assert captured["outdir"] == str(tmp_path / "ks_stationarity")
    assert captured["x_min"] == expected_x_min
