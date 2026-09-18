import math
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree

from base.boundingbox import KerogenBox, Range
from base.kerogendata import KerogenData
from utils.utils import kprint

RadiusTrees = List[Tuple[float, cKDTree]]


def _build_radius_trees(
    positions: npt.NDArray[np.float32], sizes: npt.NDArray[np.float32]
) -> RadiusTrees:
    """One cKDTree per distinct atom radius, so `is this grid point inside
    some atom's sphere` becomes a single vectorized nearest-neighbor query
    per radius group (`tree.query(..., distance_upper_bound=radius)`)
    instead of a brute-force distance scan against every atom."""
    return [
        (float(radius), cKDTree(positions[sizes == radius]))
        for radius in np.unique(sizes)
    ]


class BinarizeAlgo(Enum):
    SEQUENTIAL = "sequential"
    THREAD_SLICE = "thread_slice"
    THREAD_CHUNK = "thread_chunk"
    PROCESS_CHUNK = "process_chunk"


_proc_worker_state: Dict[str, Any] = {}


def _proc_worker_init(
    radius_trees: RadiusTrees,
    vox_sizes: "np.ndarray",
    mins: "np.ndarray",
    img_size: Tuple[int, int, int],
    yz_flat: "np.ndarray",
) -> None:
    _proc_worker_state.update(
        radius_trees=radius_trees,
        vox_sizes=vox_sizes,
        mins=mins,
        img_size=img_size,
        yz_flat=yz_flat,
    )


def _proc_worker_process_chunk(
    chunk_indices: List[int],
) -> List[Tuple[int, npt.NDArray[np.int8]]]:
    s = _proc_worker_state
    radius_trees: RadiusTrees = s["radius_trees"]
    vox_sizes = s["vox_sizes"]
    mins = s["mins"]
    img_size = s["img_size"]
    yz_flat = s["yz_flat"]
    Ny, Nz = img_size[1], img_size[2]

    results = []
    for ix in chunk_indices:
        gx = float(mins[0] + (ix + 0.5) * vox_sizes[0])
        x_col = np.full((Ny * Nz, 1), gx, dtype=np.float32)
        pos = np.hstack([x_col, yz_flat])
        atom_mask = np.zeros(Ny * Nz, dtype=bool)
        for radius, tree in radius_trees:
            dist, _ = tree.query(
                pos, k=1, distance_upper_bound=radius, workers=1
            )
            atom_mask |= dist < radius
        results.append((ix, (~atom_mask).reshape(Ny, Nz).astype(np.int8)))
    kprint(f"Chunk slices {chunk_indices[0]}..{chunk_indices[-1]}-x finished!")
    return results


class Segmentator:
    def __init__(
        self,
        kerogen: KerogenData,
        img_size: Tuple[int, int, int],
        size_data: Callable[[int], float],
        radius_extention: Callable[[int], float],
    ):
        self.kerogen = kerogen
        self.img_size = img_size
        self.radius_extention = radius_extention

        self.all_positions = np.array(
            [a.pos for a in kerogen.atoms], dtype=np.float32
        )
        self.atom_sizes = np.array(
            [
                size_data(a.type_id) + radius_extention(a.type_id)
                for a in kerogen.atoms
            ],
            dtype=np.float32,
        )
        self.radius_trees: RadiusTrees = _build_radius_trees(
            self.all_positions, self.atom_sizes
        )

    @staticmethod
    def cut_cell(
        size: Tuple[float, float, float], dev: float = 4.0
    ) -> KerogenBox:
        maxs = np.max(np.array(size))
        mins = np.min(np.array(size))
        ax_ns = min(maxs / dev, mins)
        new_cell_size = tuple(min(ax_ns, s) for s in size)
        minb = tuple((s - ns) / 2 for s, ns in zip(size, new_cell_size))
        maxb = tuple((s + ns) / 2 for s, ns in zip(size, new_cell_size))
        return KerogenBox(
            Range(minb[0], maxb[0]),
            Range(minb[1], maxb[1]),
            Range(minb[2], maxb[2]),
        )

    @staticmethod
    def full_cell(size: Tuple[float, float, float]) -> KerogenBox:
        return KerogenBox(
            Range(0.0, size[0]),
            Range(0.0, size[1]),
            Range(0.0, size[2]),
        )

    @staticmethod
    def calc_image_size(
        cell_size: Tuple[float, float, float],
        reference_size: int = 100,
        by_min: bool = True,
    ) -> Tuple[int, int, int]:
        l_cell_size = [s for s in cell_size]
        ref_cell_size = min(l_cell_size) if by_min else max(l_cell_size)
        return tuple(  # type: ignore
            int(math.ceil(reference_size * cs / ref_cell_size))
            for cs in l_cell_size
        )

    @staticmethod
    def _make_chunks(total: int, chunk_size: int) -> List[List[int]]:
        indices = list(range(total))
        return [
            indices[i : i + chunk_size] for i in range(0, total, chunk_size)
        ]

    def binarize(
        self,
        num_workers: int = 15,
        algo: Optional[BinarizeAlgo] = None,
        chunk_size: int = 8,
    ) -> npt.NDArray[np.int8]:
        """Return 3D image where 0 = atom, 1 = void."""
        if algo is None:
            algo = (
                BinarizeAlgo.SEQUENTIAL
                if num_workers == 0
                else BinarizeAlgo.THREAD_SLICE
            )

        vox_sizes = np.array(
            [
                ker_s / float(img_s)
                for ker_s, img_s in zip(self.kerogen.box.size(), self.img_size)
            ],
            dtype=np.float32,
        )
        mins = self.kerogen.box.min()
        Nx, Ny, Nz = self.img_size
        gy = (mins[1] + (np.arange(Ny) + 0.5) * vox_sizes[1]).astype(np.float32)
        gz = (mins[2] + (np.arange(Nz) + 0.5) * vox_sizes[2]).astype(np.float32)
        yy, zz = np.meshgrid(gy, gz, indexing='ij')
        yz_flat = np.stack([yy.ravel(), zz.ravel()], axis=1)  # (Ny*Nz, 2)

        img = np.ones(self.img_size, dtype=np.int8)

        def process_slice(ix: int) -> npt.NDArray[np.int8]:
            gx = float(mins[0] + (ix + 0.5) * vox_sizes[0])
            x_col = np.full((Ny * Nz, 1), gx, dtype=np.float32)
            pos = np.hstack([x_col, yz_flat])
            atom_mask = np.zeros(Ny * Nz, dtype=bool)
            for radius, tree in self.radius_trees:
                dist, _ = tree.query(
                    pos, k=1, distance_upper_bound=radius, workers=1
                )
                atom_mask |= dist < radius
            return (~atom_mask).reshape(Ny, Nz).astype(np.int8)

        if algo == BinarizeAlgo.SEQUENTIAL:
            for ix in range(Nx):
                img[ix] = process_slice(ix)
                kprint(f"Slice {ix}-x finished!")

        elif algo == BinarizeAlgo.THREAD_SLICE:
            n = min(num_workers, Nx)
            with ThreadPoolExecutor(max_workers=n) as executor:
                for ix, result in enumerate(
                    executor.map(process_slice, range(Nx))
                ):
                    img[ix] = result
                    kprint(f"Slice {ix}-x finished!")

        elif algo == BinarizeAlgo.THREAD_CHUNK:
            n = min(num_workers, Nx)
            chunks = Segmentator._make_chunks(Nx, chunk_size)

            def process_chunk_thread(
                chunk_indices: List[int],
            ) -> List[Tuple[int, npt.NDArray[np.int8]]]:
                results = [(ix, process_slice(ix)) for ix in chunk_indices]
                kprint(
                    f"Chunk slices {chunk_indices[0]}..{chunk_indices[-1]}-x finished!"
                )
                return results

            with ThreadPoolExecutor(max_workers=n) as executor:
                futures = [
                    executor.submit(process_chunk_thread, ch) for ch in chunks
                ]
                for fut in futures:
                    for ix, result in fut.result():
                        img[ix] = result

        elif algo == BinarizeAlgo.PROCESS_CHUNK:
            n = min(num_workers, Nx)
            chunks = Segmentator._make_chunks(Nx, chunk_size)
            with ProcessPoolExecutor(
                max_workers=n,
                initializer=_proc_worker_init,
                initargs=(
                    self.radius_trees,
                    vox_sizes,
                    mins,
                    self.img_size,
                    yz_flat,
                ),
            ) as process_executor:
                futures = [
                    process_executor.submit(_proc_worker_process_chunk, ch)
                    for ch in chunks
                ]
                for fut in futures:
                    for ix, result in fut.result():
                        img[ix] = result

        return img

    def benchmark_binarize(
        self,
        num_workers: int = 8,
        chunk_size: int = 8,
        algos: Optional[List[BinarizeAlgo]] = None,
    ) -> Dict[str, float]:
        """Run binarize() with each algorithm variant and print a timing comparison."""
        if algos is None:
            algos = list(BinarizeAlgo)
        results: Dict[str, float] = {}
        for algo in algos:
            kprint(f"[benchmark] Starting {algo.value} ...")
            t0 = time.perf_counter()
            self.binarize(
                num_workers=num_workers,
                algo=algo,
                chunk_size=chunk_size,
            )
            results[algo.value] = time.perf_counter() - t0
            kprint(f"[benchmark] {algo.value}: {results[algo.value]:.3f}s")

        baseline = results.get(BinarizeAlgo.SEQUENTIAL.value)
        kprint("Benchmark Results")
        for name, t in results.items():
            speedup = f"  ({baseline / t:.2f}x)" if baseline and t > 0 else ""
            kprint(f"    {name:<20s}: {t:8.3f}s{speedup}")
        return results

    def dist_map(self) -> npt.NDArray[np.float32]:
        vox_sizes = [
            ker_s / float(img_s)
            for ker_s, img_s in zip(self.kerogen.box.size(), self.img_size)
        ]

        tree = cKDTree(self.all_positions)

        def wrap(ix: int) -> npt.NDArray[np.float32]:
            ny, nz = self.img_size[1], self.img_size[2]
            box_min = self.kerogen.box.min()
            iy_idx = np.arange(ny, dtype=np.float32)
            iz_idx = np.arange(nz, dtype=np.float32)
            IY, IZ = np.meshgrid(iy_idx, iz_idx, indexing='ij')
            all_pos = np.column_stack(
                [
                    np.full(
                        ny * nz,
                        box_min[0] + (ix + 0.5) * vox_sizes[0],
                        dtype=np.float32,
                    ),
                    (box_min[1] + (IY + 0.5) * vox_sizes[1]).ravel(),
                    (box_min[2] + (IZ + 0.5) * vox_sizes[2]).ravel(),
                ]
            )  # (Ny*Nz, 3)
            dist, _ = tree.query(all_pos, k=1, workers=1)
            return cast(
                npt.NDArray[np.float32],
                dist.astype(np.float32).reshape(ny, nz),
            )

        img = np.ones(shape=self.img_size, dtype=np.float32)
        for ix in range(self.img_size[0]):
            img[ix] = wrap(ix)
            kprint(f"Slice {ix}-x finished!")

        return img
