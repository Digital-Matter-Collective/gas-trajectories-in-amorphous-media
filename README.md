# gas-trajectories-in-amorphous-media

gas-trajectories-in-amorphous-media is a research package for studying gas-molecule trajectories
in dynamic kerogen pore structures. It implements trapping-event classifiers,
synthetic validation experiments, molecular-dynamics trajectory preprocessing,
and downstream pore-network statistics used in the accompanying research.

The code and the accompanying research-data archive (molecular-dynamics
trajectories and reference pore-network models, PNMs, for CH4/H2 in type-I
kerogen at 300 K) are both published on Zenodo; see "Citation and license"
below for their concept DOIs and the exact records for release `v1.1.0`.

## Scientific scope

The package separates a gas trajectory into free and trapped steps and compares
several classification approaches:

- distance-matrix (DM) analysis;
- Neyman-Pearson (NP) analysis;
- NP initialization followed by Bayesian refinement (SIB);
- a hybrid DM and Bayesian classifier (HYB).

It also contains tools for extracting gas and kerogen trajectories from
GRO-like molecular-dynamics files, sampling dynamic kerogen structures,
building binary images and distance maps, reading Statoil-format PNMs, and
summarizing trapping and stationarity statistics.

All trajectory analyzers use an edge-label contract: a trajectory with
\(N\) points produces \(N-1\) Boolean labels. The DM classifier first labels
points and assigns each edge the label of its destination point.

## Requirements and installation

The reference environment is Ubuntu x86-64, Python 3.14.4, and the MKL
scientific stack pinned in [`environment.yml`](environment.yml). Create it
from the repository root:

```bash
CONDA_CHANNEL_PRIORITY=flexible conda env create -f environment.yml
conda activate kerogen
```

The environment installs the project in editable mode, including tests,
bundled DM threshold tables, and optional VTK visualization. Verify it with:

```bash
python -m pytest -q
gas-traj-dm-search --help
```

On the reference workstation (Ubuntu x86-64, AMD Ryzen 7 5800H), a clean
installation of this complete environment took 119 seconds on 28 September
2026. The time includes dependency solving, downloads, extraction, and the
editable package installation; allow additional time on a slower network or
when the Conda package cache is empty.

A regular pip installation is supported for development, but is not the
reference environment for reproducing numerical results:

```bash
python -m pip install ".[visualization,test]"
```

## Quality checks

`prepare.sh` runs the local autofix pass (Ruff autofix, isort, Black, a
final Ruff check, mypy) over every tracked `*.py` file:

```bash
bash prepare.sh
```

`tools/check.sh` runs the same tools in non-mutating `--check` mode plus
the full test suite, and is exactly what CI (`.github/workflows/ci.yml`)
runs on every push and pull request:

```bash
bash tools/check.sh
```

Both scripts share the same file list (`tools/list_python_files.sh`,
tracked files only), so they can never check a different set of files
than each other.

## Quick start

This self-contained example classifies a seeded three-dimensional random walk
with the current optimized DM parameter profile for \(k=0.5\). It needs no
publication data:

```bash
python - <<'PY'
import numpy as np

from base.boundingbox import BoundingBox, Range
from base.trajectory import Trajectory
from processes.trajectory_analyzer.dm import (
    DistanceMatrixAnalyzer,
    DistanceMatrixParams,
)

rng = np.random.default_rng(42)
points = (
    50.0 + np.cumsum(rng.normal(0.0, 0.02, (100, 3)), axis=0)
).astype(np.float32)
trajectory = Trajectory(
    points=points,
    times=np.arange(100, dtype=np.float32),
    box=BoundingBox(Range(0, 100), Range(0, 100), Range(0, 100)),
)
params = DistanceMatrixParams(
    traj_type="fBm",
    nu=0.5,
    diag_percentile=0,
    kernel_size=3,
    list_mu=np.array([1.5, 2.0, 2.5], dtype=np.float32),
    p_value=0.9,
)
labels = DistanceMatrixAnalyzer(params).run(trajectory)
print(
    f"points={trajectory.count_points}, "
    f"edge_labels={labels.size}, trapped={labels.sum()}"
)
PY
```

Expected output:

```text
points=100, edge_labels=99, trapped=85
```

On the reference workstation, the first run through `conda run -n kerogen`
took 3.64 seconds, including process and environment startup.

## Command-line workflow

Installing the package provides these supported entry points:

| Stage | Command | Main output |
|---|---|---|
| Filter a gas trajectory | `gas-traj-extract-gas-trajectory` | GRO-like trajectory |
| Filter selected kerogen molecules | `gas-traj-extract-krg-trajectory` | GRO-like trajectory |
| Extract dynamic structures | `gas-traj-extract-structures` | structure `.npz` files |
| Binarize structures | `gas-traj-binarize-structures` | NumPy and raw images |
| Build distance maps | `gas-traj-distance-maps` | NumPy distance maps |
| Search DM parameters (Table IV) | `gas-traj-dm-search` | checkpoints and search manifest |
| Aggregate DM search (Table IV) | `gas-traj-find-dm-params` | CSV and JSON |
| Run synthetic validation (Figs. 8, 13; Table I) | `gas-traj-synthetic-benchmark` | SVG, CSV, JSON, manifest |
| Build trapping distributions (Table II) | `gas-traj-trap-distributions` | SVG, CSV, JSON |
| Analyze PNM stationarity (Table III) | `gas-traj-stationarity` | SVG, CSV, JSON |
| Build or verify a data manifest | `gas-traj-data-manifest` | SHA-256 JSON manifest |

Use `COMMAND --help` for the complete interface. The data preparation
sequence, publication profiles, expected file layout, and figure/table mapping
are documented in [`docs/reproduction.md`](docs/reproduction.md). That guide
also states the console-script-vs-`python -m` convention and documents the
remaining manual/visualization scripts under `scripts/` and the PyMOL/VMD
renderers under `external_scripts/`, which are not installed as commands. A
description of every positional argument and option, including defaults and
units, is in the
[`command-line parameter reference`](docs/cli-reference.md).

### Manual and visualization scripts

Beyond the installed commands above, `scripts/` also contains one-off
plotters and interactive VTK/PyMOL/VMD visualization entry points (structure
overlays, trajectory renderers, correlation-function and PNM plots). They are
run as `python -m scripts.<module>`, not installed as console scripts. Full
copy-ready commands are in the
[reproduction guide](docs/reproduction.md#10-manual-structure-and-trajectory-visualization-figures-1-2-4-7-10).

## Input data

The preprocessing commands accept multi-frame GROMACS GRO-like trajectories.
Headers must contain simulation time and step metadata used to align
trajectories, structures, and PNMs.

PNM analysis expects one Statoil-format network per common `<prefix>`:

- `<prefix>_node1.dat`: pore-center coordinates;
- `<prefix>_node2.dat`: pore properties, including radius;
- `<prefix>_link1.dat`: connected pore pairs and throat properties;
- `<prefix>_link2.dat`: additional standard link properties.

The current reader uses `node1`, `node2`, and `link1`. Lengths in PNM
files are expected in metres and are converted to nanometres.

The authors' PNM extractor is a separate closed implementation and is not part
of this repository. [`scripts/pnm_extractor.py`](scripts/pnm_extractor.py) is
an adapter example, not a bundled extractor. Another extractor may be used if
its output conforms to the schema above, but it may produce numerically
different networks.

Exact downstream reproduction does not require access to that extractor. The
published dataset record
<https://doi.org/10.5281/zenodo.22899527> contains the authors' reference
networks as `ch4_pnm.tar.gz` and `h2_pnm.tar.gz` (100 PNMs and 400
Statoil-format files per gas), together with the input trajectories,
intermediate data, classifier outputs, and derived summaries. Download only
the per-gas archives needed for a particular workflow; extraction and command
examples are given in [`docs/reproduction.md`](docs/reproduction.md).

## Outputs, runtime, and reproducibility

Publication-facing commands write machine-readable CSV/JSON summaries next to
figures and record seeds or run parameters where implemented. Long-running
searches use checkpoints; read each command's `--help` before replacing or
resuming cached results.

Runtime depends strongly on trajectory length, number of molecules and frames,
image resolution, and CPU count. In particular, the DM method constructs
quadratic-size distance matrices. The full Table IV grid and publication
synthetic benchmark are research calculations, not quick-start workloads.
Allow for Numba compilation on the first call and monitor memory before scaling
up. The reference workflow is CPU-based; no GPU is required.

## Troubleshooting

- If Conda reports channel conflicts, keep
  `CONDA_CHANNEL_PRIORITY=flexible` when creating the reference environment.
- If Matplotlib or Fontconfig cannot write a cache in a restricted session,
  point `MPLCONFIGDIR` and `XDG_CACHE_HOME` to writable directories.
- VTK is needed only for optional interactive visualization. Headless analysis
  and tests do not require an open display.
- A missing `pi_l_gamma_fitter.json`, PNM prefix, trajectory, or cached
  distribution means that the requested data-dependent stage has not been
  prepared; follow the input sequence in the reproduction guide.
- Do not mix old caches with changed inputs or parameters. Preserve current
  results before using a force-recompute option.

## Citation and license

For reproducibility, cite the exact software version recorded in
[`CITATION.cff`](CITATION.cff):

> Ananev, A., Potapova, M., Vostroknutov, T., Kondratyuk, N., & Khlyupin, A.
> (2026). *gas-trajectories-in-amorphous-media* (Version 1.1.0) [Computer software].
> Zenodo. <https://doi.org/10.5281/zenodo.22900852>

The concept DOI <https://doi.org/10.5281/zenodo.22309111> represents the
software family and resolves to its latest published version. Use it when a
version-independent link is intended; use the version DOI above when recording
the exact code used for an analysis. For future releases, reserve a version DOI
before publication or update `CITATION.cff` after Zenodo mints it.

GitHub's **Cite this repository** menu reads `CITATION.cff` and can export
this citation in APA and BibTeX formats. The software is distributed under
the [MIT License](LICENSE).

Research datasets are not distributed by this repository and are not covered
by the software license. If you use the accompanying molecular-dynamics or
derived data, cite the immutable dataset version used by this release and the
manuscript: <https://doi.org/10.5281/zenodo.22899527>. Its concept DOI,
<https://doi.org/10.5281/zenodo.22309333>, represents all dataset versions and
resolves to the latest one. Use the **Citation** panel on the Zenodo record to
export the dataset citation in the style required by the journal.

When the accompanying article receives a DOI, add it to the commented
`preferred-citation` template at the bottom of `CITATION.cff`. Use the bare
DOI (for example, `10.xxxx/example`), not the `https://doi.org/` URL; citation
tools will create the link automatically.
