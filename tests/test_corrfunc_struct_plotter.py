from pathlib import Path

import numpy as np
import pytest

from scripts.corrfunc_struct_plotter import correlation_average_time


def _reference_correlation(images: list[np.ndarray]) -> np.ndarray:
    result = np.empty(len(images), dtype=np.float64)
    for lag in range(len(images)):
        values = [
            np.sum(images[i] * images[i + lag]) / np.sum(images[i])
            for i in range(len(images) - lag)
        ]
        result[lag] = np.mean(values)
    return result


@pytest.mark.parametrize("num_workers", [0, 1, 3])
def test_correlation_average_time_matches_dense_reference(
    tmp_path: Path, num_workers: int
) -> None:
    rng = np.random.default_rng(42)
    images = [
        rng.integers(0, 2, size=(5, 4, 7), dtype=np.int8) for _ in range(6)
    ]
    image_infos = [(float(i * 10), str(i)) for i in range(len(images))]

    dt, actual = correlation_average_time(
        image_infos,
        lambda name: images[int(name)],
        tmp_path / f"ct-{num_workers}.npy",
        num_workers=num_workers,
    )

    np.testing.assert_allclose(actual, _reference_correlation(images))
    np.testing.assert_allclose(dt, np.arange(len(images)) * 1e-5)


def test_correlation_does_not_load_images_for_complete_cache(
    tmp_path: Path,
) -> None:
    images = [
        np.array([1, 0, 1, 0, 1], dtype=np.int8),
        np.array([1, 1, 0, 0, 1], dtype=np.int8),
        np.array([0, 1, 1, 0, 1], dtype=np.int8),
    ]
    image_infos = [(float(i), str(i)) for i in range(len(images))]
    cache_path = tmp_path / "ct.npy"

    correlation_average_time(
        image_infos,
        lambda name: images[int(name)],
        cache_path,
        num_workers=0,
    )

    load_calls = 0

    def load_image(name: str) -> np.ndarray:
        nonlocal load_calls
        load_calls += 1
        return images[int(name)]

    _, resumed = correlation_average_time(
        image_infos,
        load_image,
        cache_path,
        num_workers=2,
    )

    np.testing.assert_allclose(resumed, _reference_correlation(images))
    assert load_calls == 0


def test_correlation_resumes_partial_cache(tmp_path: Path) -> None:
    images = [
        np.array([1, 0, 1, 0, 1], dtype=np.int8),
        np.array([1, 1, 0, 0, 1], dtype=np.int8),
        np.array([0, 1, 1, 0, 1], dtype=np.int8),
    ]
    image_infos = [(float(i), str(i)) for i in range(len(images))]
    cache_path = tmp_path / "ct.npy"
    expected = _reference_correlation(images)

    correlation_average_time(
        image_infos,
        lambda name: images[int(name)],
        cache_path,
        num_workers=0,
    )
    np.save(cache_path, np.array([expected[0], np.nan, np.nan]))

    load_calls = 0

    def load_image(name: str) -> np.ndarray:
        nonlocal load_calls
        load_calls += 1
        return images[int(name)]

    _, resumed = correlation_average_time(
        image_infos,
        load_image,
        cache_path,
        num_workers=2,
    )

    np.testing.assert_allclose(resumed, expected)
    assert load_calls == len(images)


def test_correlation_rejects_different_image_shapes(tmp_path: Path) -> None:
    images = [
        np.ones((2, 3), dtype=np.int8),
        np.ones((3, 2), dtype=np.int8),
    ]
    image_infos = [(float(i), str(i)) for i in range(len(images))]

    with pytest.raises(ValueError, match="has shape"):
        correlation_average_time(
            image_infos,
            lambda name: images[int(name)],
            tmp_path / "ct.npy",
            num_workers=0,
        )
