import numpy as np

from base.boundingbox import BoundingBox, Range
from base.kerogendata import AtomData, KerogenData
from processes.segmentation import BinarizeAlgo, Segmentator

# Two distinct radii on purpose: the KD-tree implementation groups atoms by
# radius (processes/segmentation.py:_build_radius_trees), so a fixture with
# only one radius would not exercise that grouping at all.
_RADII = {0: 0.3, 1: 0.1}


def _size_data(type_id: int) -> float:
    return _RADII[type_id]


def _zero_extension(type_id: int) -> float:
    return 0.0


def _make_segmentator(
    img_size: tuple[int, int, int] = (10, 10, 10)
) -> Segmentator:
    box = BoundingBox(Range(0.0, 2.0), Range(0.0, 2.0), Range(0.0, 2.0))
    atoms = [
        AtomData(
            0, "KRG", "C0", 0, np.array([1.0, 1.0, 1.0], dtype=np.float32)
        ),
        AtomData(
            1, "KRG", "C1", 1, np.array([1.0, 1.0, 1.8], dtype=np.float32)
        ),
        AtomData(
            2, "KRG", "C2", 1, np.array([0.2, 0.2, 0.2], dtype=np.float32)
        ),
    ]
    kerogen = KerogenData(None, atoms, box)
    return Segmentator(
        kerogen,
        img_size,
        size_data=_size_data,
        radius_extention=_zero_extension,
    )


def _brute_force_reference(segmentator: Segmentator) -> np.ndarray:
    """Independent ground truth: for every grid point, check every atom
    directly. O(N_grid * N_atoms), fine for this tiny fixture; used to
    verify the KD-tree implementation without depending on it."""
    Nx, Ny, Nz = segmentator.img_size
    vox_sizes = [
        s / n for s, n in zip(segmentator.kerogen.box.size(), (Nx, Ny, Nz))
    ]
    mins = segmentator.kerogen.box.min()
    img = np.ones((Nx, Ny, Nz), dtype=np.int8)
    for ix in range(Nx):
        for iy in range(Ny):
            for iz in range(Nz):
                point = np.array(
                    [
                        mins[0] + (ix + 0.5) * vox_sizes[0],
                        mins[1] + (iy + 0.5) * vox_sizes[1],
                        mins[2] + (iz + 0.5) * vox_sizes[2],
                    ]
                )
                dists = np.linalg.norm(
                    segmentator.all_positions - point, axis=1
                )
                inside = np.any(dists < segmentator.atom_sizes)
                img[ix, iy, iz] = 0 if inside else 1
    return img


def test_binarize_matches_a_brute_force_reference_with_mixed_atom_radii() -> (
    None
):
    segmentator = _make_segmentator()
    expected = _brute_force_reference(segmentator)

    actual = segmentator.binarize(num_workers=0, algo=BinarizeAlgo.SEQUENTIAL)

    np.testing.assert_array_equal(actual, expected)
    # Sanity check the fixture actually exercises both "inside" and
    # "outside" voxels, not a degenerate all-void or all-atom result.
    assert 0 in expected
    assert 1 in expected


def test_binarize_process_chunk_matches_sequential() -> None:
    segmentator = _make_segmentator()

    sequential = segmentator.binarize(
        num_workers=0, algo=BinarizeAlgo.SEQUENTIAL
    )
    process_chunk = segmentator.binarize(
        num_workers=2, algo=BinarizeAlgo.PROCESS_CHUNK, chunk_size=3
    )

    np.testing.assert_array_equal(process_chunk, sequential)


def test_binarize_thread_chunk_matches_sequential() -> None:
    segmentator = _make_segmentator()

    sequential = segmentator.binarize(
        num_workers=0, algo=BinarizeAlgo.SEQUENTIAL
    )
    thread_chunk = segmentator.binarize(
        num_workers=2, algo=BinarizeAlgo.THREAD_CHUNK, chunk_size=3
    )

    np.testing.assert_array_equal(thread_chunk, sequential)
