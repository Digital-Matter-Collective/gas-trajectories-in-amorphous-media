import json
from pathlib import Path

from scripts.build_data_release_manifest import (
    build_manifest,
    save_manifest,
    verify_manifest,
)


def _make_data_dir(tmp_path: Path) -> Path:
    data_dir = tmp_path / "release"
    (data_dir / "pnm").mkdir(parents=True)
    (data_dir / "pnm" / "sample_node1.dat").write_text("1 2 3\n")
    (data_dir / "readme.txt").write_text("reference PNM outputs\n")
    return data_dir


def test_build_manifest_hashes_every_file_and_skips_itself(
    tmp_path: Path,
) -> None:
    data_dir = _make_data_dir(tmp_path)

    manifest = build_manifest(
        data_dir,
        license_name="CC-BY-4.0",
        code_url="https://github.com/Digital-Matter-Collective/gas-trajectories-in-amorphous-media",
        code_version="v1.0.0",
        description="Reference PNM outputs",
    )

    assert manifest["file_count"] == 2
    assert manifest["license"] == "CC-BY-4.0"
    paths = {entry["path"] for entry in manifest["files"]}
    assert paths == {"pnm/sample_node1.dat", "readme.txt"}

    manifest_path = save_manifest(manifest, data_dir)
    assert manifest_path.name == "data_manifest.json"
    with manifest_path.open() as file:
        assert json.load(file)["file_count"] == 2

    rebuilt = build_manifest(
        data_dir,
        license_name="CC-BY-4.0",
        code_url=None,
        code_version=None,
        description=None,
    )
    assert rebuilt["file_count"] == 2, "manifest file itself must not be hashed"


def test_verify_manifest_detects_missing_corrupted_and_unexpected_files(
    tmp_path: Path,
) -> None:
    data_dir = _make_data_dir(tmp_path)
    manifest = build_manifest(
        data_dir,
        license_name="CC-BY-4.0",
        code_url=None,
        code_version=None,
        description=None,
    )
    manifest_path = save_manifest(manifest, data_dir)

    missing, corrupted, unexpected = verify_manifest(data_dir, manifest_path)
    assert (missing, corrupted, unexpected) == ([], [], [])

    (data_dir / "readme.txt").write_text("tampered\n")
    (data_dir / "pnm" / "sample_node1.dat").unlink()
    (data_dir / "extra.dat").write_text("not in manifest\n")

    missing, corrupted, unexpected = verify_manifest(data_dir, manifest_path)
    assert missing == ["pnm/sample_node1.dat"]
    assert corrupted == ["readme.txt"]
    assert unexpected == ["extra.dat"]
