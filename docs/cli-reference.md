# Command-line parameter reference

This reference covers every executable script used in the reproduction guide.
Usage examples and workflow order remain in
[`reproduction.md`](reproduction.md); this document explains every accepted
parameter.

Run Python scripts from the repository root as `python -m scripts.<name>`.
Where `pyproject.toml` provides a `gas-traj-*` command, that command accepts
the same parameters. Every `argparse`-based Python command also accepts
`-h`/`--help`, which prints its current interface and exits. Paths may be
relative to the repository root or absolute. “Off” means that a Boolean flag
is false unless it is present.

## Trajectory and structure preparation

### `extract_gas_trajectory_to_file`

Installed command: `gas-traj-extract-gas-trajectory`.

| Parameter | Required/default | Meaning |
|---|---|---|
| `input` | required | Input multi-frame GRO-like trajectory. |
| `output` | required | Destination GRO-like trajectory. |
| `--exclude-resname NAME` | optional; `KRG` | Residue name to remove. Repeat the option to remove several names. Supplying any value replaces the default exclusion set. |
| `--keep-original-count` | off | Preserve each input atom-count line even when atoms were removed. This normally makes the filtered frame inconsistent, so omit it for valid GRO-like output. |
| `--renumber-atoms` | off | Renumber atom serials in every filtered frame. |
| `--drop-empty-frames` | off | Omit frames in which filtering leaves no atoms; otherwise retain empty frames. |

### `extract_krg_trajectory_to_file`

Installed command: `gas-traj-extract-krg-trajectory`.

| Parameter | Required/default | Meaning |
|---|---|---|
| `input` | required | Input multi-frame GRO-like trajectory. |
| `output` | required | Destination filtered trajectory. |
| `--select RESNAME:NUMBER` | required, repeatable | Keep the specified residue/molecule number, for example `KRG:99`. Repeat to keep several selections. |
| `--keep-original-count` | off | Preserve the original atom-count lines; normally omit this so counts match the filtered frames. |
| `--renumber-atoms` | off | Renumber atom serials in every output frame. |
| `--drop-empty-frames` | off | Omit frames containing none of the requested selections. |

### `dynamic_struct_extractor`

Installed command: `gas-traj-extract-structures`.

| Parameter | Required/default | Meaning |
|---|---|---|
| `input` | required | Input multi-frame GRO trajectory. |
| `output_dir` | required | Directory for extracted `struct-num=..._time-ps=....npz` structures; it is created if needed. |
| `--index STEP` | selection required unless `--auto-indexes`; repeatable | Extract a simulation step. One occurrence may contain comma- or whitespace-separated steps. A request for the first trajectory frame is ignored. |
| `--indexes-file FILE` | selection required unless `--auto-indexes` | Text file containing comma- or whitespace-separated simulation steps. It may be combined with `--index`. |
| `--auto-indexes` | off | Read available steps from the GRO headers, always exclude the first frame, and select the requested structures. Mutually exclusive with `--index`/`--indexes-file`. |
| `--mode {all,part}` | `all` | With `--auto-indexes`, `all` spreads steps from the second available frame through the last, including both endpoints; `part` takes frames consecutively beginning with the second. |
| `--count-structures N` | required with `--auto-indexes` | Number of structures requested. Fewer are returned only if fewer frames remain after excluding the first. |
| `--slice-len N` | `100` | Number of requested structures read in one batch; affects I/O batching, not the selected set. |
| `--dry-run` | off | Print the generated/explicit indexes without extracting structures. |

### `binarization_structs`

Installed command: `gas-traj-binarize-structures`.

| Parameter | Required/default | Meaning |
|---|---|---|
| `structures_dir` | required | Directory containing extracted structure `.npz` files. |
| `output_bin_dir` | required | Directory for binary `.npy` volumes. |
| `output_raw_dir` | required | Directory for matching headerless `.raw` volumes. |
| `--ref-size N` | required | Voxel count along the shortest side of the cropped bounding box. It determines spatial resolution and has roughly quadratic influence on per-slice working memory. |
| `--dev FLOAT` | `2.0` | Cell-cropping divisor passed to `Segmentator.cut_cell`; larger values retain a smaller central box. |
| `--num-workers N` | `4` | Number of worker processes used to binarize slices. More workers increase both concurrency and peak memory. |
| `--atom-chunk N` | `1024` | Atoms per pairwise-distance batch in each worker. Lower values reduce peak memory at some CPU cost. |

Every extracted structure `.npz` file in `structures_dir` is processed.

### `distance_map_structs`

Installed command: `gas-traj-distance-maps`.

| Parameter | Required/default | Meaning |
|---|---|---|
| `structures_dir` | required | Directory containing extracted structure `.npz` files. |
| `output_dir` | required | Base output directory; maps are written below `float_images/`. |
| `--ref-size N` | required | Voxel count along the cropped box’s shortest side and therefore the map resolution control. |
| `--dev FLOAT` | `4.0` | Cell-cropping divisor; larger values retain a smaller central box. |

Every extracted structure `.npz` file in `structures_dir` is processed.

### `pnm_extractor`

This adapter has no installed command.

| Parameter | Required/default | Meaning |
|---|---|---|
| `path` | required | Data directory containing `raw_images/`; generated networks are placed under its `pnm/` directory. |
| `extractor` | required | Path to the external PNM-extractor executable, which is not included in this repository. |
| `config` | required | JSON configuration passed to the external extractor. |

### `generate_pil_distr`

Installed command: `gas-traj-generate-pil-distr`.

| Parameter | Required/default | Meaning |
|---|---|---|
| `pnm_dir` | required | Input directory of Statoil-format PNMs. Each network prefix needs `_node2.dat` and `_link1.dat`. |
| `output_dir` | required | Existing directory for NumPy samples, fitted-distribution JSON, provenance metadata, and `figs/`. |
| `--x-min FLOAT` | `0.025` nm | Keep only pore radii strictly above this threshold when generating/fitting the pore-intersection-length sample. It does not filter the saved raw `radiuses.npy` or the throat-length fit. |

## Publication calculations

### `errors_params`

Installed command: `gas-traj-dm-search`.

| Parameter | Required/default | Meaning |
|---|---|---|
| `path` | required | Data directory containing `radiuses.npy` and `throat_lengths_weibull_fitter.json`; results go to `errors/find_best_params/`. |
| `--trajectory-count N` | `100` | Synthetic trajectories evaluated for every `(k,p)` pair. Lower values are useful for experiments but do not reproduce Table IV. |
| `--trajectory-points N` | `1000` | Points in each synthetic trajectory; must be at least 2. DM cost grows approximately quadratically with this value. |
| `--seed N` | `42` | Base seed for deterministic NumPy and Python random streams. Must match existing checkpoints. |
| `--n-jobs N` | `-1` | Joblib worker count; `-1` uses all available logical CPUs and `0` is invalid. |
| `--force-recompute` | off | Ignore compatible checkpoints and recalculate every `(k,p)` pair. Mutually exclusive with `--resume`. |
| `--resume` | off | Accept compatible scale checkpoints and permit old metadata-free result files to be replaced after their first new checkpoint. Mutually exclusive with `--force-recompute`. |

### `find_best_params`

Installed command: `gas-traj-find-dm-params`.

| Parameter | Required/default | Meaning |
|---|---|---|
| `path` | required | Directory containing the completed `k=..._p=....npz` Table IV search results. |
| `--output-dir DIR` | input `path` | Directory for `optimized_dm_params.csv` and `.json`. |

### `sim_algo_check`

Installed command: `gas-traj-synthetic-benchmark`.

| Parameter | Required/default | Meaning |
|---|---|---|
| `path` | required | Data directory containing the fitted distribution inputs; figures, caches, manifests, and Table II are written below it. |
| `--trajectory-count N` | `100` | Synthetic trajectories per tested `(k,p)` combination. |
| `--step-count N` | `3000` | Simulated steps per trajectory. |
| `--seed N` | `42` | Base random seed recorded in cache/checkpoint provenance. |
| `--force-recompute` | off | Replace trajectory caches and all analyzer checkpoints instead of reusing compatible results. |

### `trap_distr_builder`

Installed command: `gas-traj-trap-distributions`.

| Parameter | Required/default | Meaning |
|---|---|---|
| `path` | required | Data directory containing `trj.gro` and fitted PIL/throat distributions. |
| `--label LABEL` | required | Gas name stored in the Table II rows and used for plot ranges, normally `CH4` or `H2`. Unknown labels use CH4 plot limits. |
| `--num N` | `1` | Molecule stride: analyze every Nth trajectory/molecule from `trj.gro`. |
| `--output FILE` | `<path>/traps/P(t)_loglog.svg` | SVG destination for the trapping-time plot. |
| `--recompute {DM,SIB,HYB}` | optional, repeatable | Recalculate the named analyzer while reusing compatible caches for the others. |
| `--force-recompute` | off | Recalculate all DM, SIB, and HYB label/sequence caches; takes precedence over `--recompute`. |
| `--summary-dir DIR` | `<path>/traps` | Directory for shared Table II CSV/JSON. Pass the same directory for different gases to merge their rows. |

### `stationarity`

Installed command: `gas-traj-stationarity`.

| Parameter | Required/default | Meaning |
|---|---|---|
| `path` | required | Data directory containing `pnm/`; output is written to `ks_stationarity/`. |
| `--trajectory FILE` | `<path>/trj.gro` | GRO trajectory whose first two frames provide an affine simulation-step/time mapping. |
| `--anchor-step N` | inferred | Simulation step at the mapping anchor. |
| `--anchor-time-ps FLOAT` | inferred | Time in ps at the mapping anchor. |
| `--step-delta N` | inferred | Simulation-step difference between two frames. |
| `--time-delta-ps FLOAT` | inferred | Time difference in ps corresponding to `--step-delta`. |

The four mapping values may override individual values inferred from the
trajectory. If the trajectory is unavailable, all four must be supplied.

### `build_data_release_manifest`

Installed command: `gas-traj-data-manifest`. It requires one subcommand.

`build` parameters:

| Parameter | Required/default | Meaning |
|---|---|---|
| `data_dir` | required | Directory recursively hashed for the release manifest. The manifest itself and its temporary file are excluded. |
| `--license SPDX` | `CC-BY-4.0` | SPDX identifier stored as the data-release license. |
| `--code-url URL` | unset | Repository or release URL stored in the manifest. |
| `--code-version TEXT` | unset | Code tag, release, or commit hash stored in the manifest. |
| `--description TEXT` | unset | Free-form dataset description stored in the manifest. |
| `--manifest-name NAME` | `data_manifest.json` | Manifest filename inside `data_dir`. |

`verify` parameters:

| Parameter | Required/default | Meaning |
|---|---|---|
| `data_dir` | required | Directory whose files are checked. |
| `--manifest FILE` | `<data_dir>/data_manifest.json` | Manifest to verify against. Missing, changed, and unexpected files produce exit status 1. |

## Analysis and plotting scripts

### `complexity_estimation`

| Parameter | Required/default | Meaning |
|---|---|---|
| `path` | required | Data directory containing `radiuses.npy` and both fitted-distribution JSON files. The timing cache is `<path>/times.npy`. |
| `output` | required | Output PDF path. |
| `--min-length N` | `500` | First trajectory length included in `numpy.arange`. |
| `--max-length N` | `8000` | Exclusive upper bound for trajectory lengths. |
| `--length-step N` | `500` | Increment between tested trajectory lengths. |
| `--repeats N` | `5` | Independent trajectories timed for each analyzer and length. |
| `--seed N` | `42` | Seed controlling the benchmark trajectories and timing-cache provenance. |

### `corrfunc_krg_mol_plotter`

| Parameter | Required/default | Meaning |
|---|---|---|
| `input` | required | Extracted kerogen-molecule GRO trajectory. |
| `output` | required | Output correlation/RMSD figure path. |
| `msd` | required | `.npz` path used to save or reuse the provenance-tracked RMSD cache. |

### `corrfunc_struct_plotter`

| Parameter | Required/default | Meaning |
|---|---|---|
| `images_dir` | required | Directory of binarized structure `.npy` images. |
| `ct_file` | required | Incremental C(t) cache path (`.npy`). |
| `output` | required | Output plot path, such as an SVG. |
| `--trj PATH:LABEL` | required, repeatable | Gas GRO trajectory and legend label. Repeat to overlay gases. |
| `--max-t FLOAT` | `2.8` µs | Maximum time used when plotting the trajectory RMSD series. |
| `--x-max FLOAT` | automatic | Explicit right-hand X-axis limit in µs. |
| `--num-workers N` | `4` | Thread count for incremental C(t) computation. |

### `distr_pnm_element_size_plotter`

| Parameter | Required/default | Meaning |
|---|---|---|
| `pnm_dir` | required | Directory containing time-indexed PNM files. |
| `figs_dir` | required | Directory in which `fill_elev_22_azim_-72.svg` is saved. |
| `--label LABEL` | empty | Series label printed in diagnostics/legend. |
| `--pnm-step N` | `100` | Take every Nth PNM in sorted order, starting after the earliest N entries; this is a file-list stride, not a simulation-step value. |
| `--bins N` | `50` | Histogram-bin count for pore-radius and throat-length densities. |
| `--no-smooth` | off | Disable Savitzky–Golay smoothing of histogram densities. |
| `--x-min FLOAT` | `0.025` nm | Strict lower X bound retained in displayed densities. |
| `--x-max FLOAT` | `2.0` nm | Strict upper X bound retained in displayed densities. |
| `--trajectory FILE` | `<pnm_dir>/../trj.gro` | GRO source used to infer PNM step-to-time conversion. |
| `--anchor-step N` | inferred | Explicit mapping anchor step; same rules as `stationarity`. |
| `--anchor-time-ps FLOAT` | inferred | Explicit mapping anchor time in ps. |
| `--step-delta N` | inferred | Explicit simulation-step interval. |
| `--time-delta-ps FLOAT` | inferred | Time in ps corresponding to the explicit step interval. |

### `pil_plotter`

| Parameter | Required/default | Meaning |
|---|---|---|
| `path` | required | Data directory containing `radiuses.npy`, `pi_l_data.npy`, fitter JSON, and `pnm_distribution_units.json`; plots go to `figs/`. |
| `--x-min FLOAT` | `0.025` nm | Minimum radius used while fitting/displaying P(r). |
| `--r-points N` | `900` | Number of radius points in the fitted grid used for the heatmap. |
| `--r-step N` | `1` | Stride through that radius grid when building conditional PIL curves. |
| `--l-bins N` | `300` | Number of length-grid nodes used for the heatmap. |
| `--heatmap-mode {smooth,mesh}` | `smooth` | `smooth` embeds a rasterized image layer; `mesh` uses vector `pcolormesh`. |
| `--heatmap-interpolation NAME` | `bicubic` | Matplotlib image interpolation in `smooth` mode. |
| `--heatmap-dpi N` | `600` | DPI of raster layers embedded in SVG and of PNG output. |
| `--fit-sample-stride N` | `10` | Fit P(r) from every Nth uniformly ordered radius; `1` uses all radii and costs more. |

### `powerlaw_analysis`

| Parameter | Required/default | Meaning |
|---|---|---|
| `path` | required | Data directory containing `traps/<prefix>/seq_*.npz`. |
| `--prefix {DM,SIB,HYB}` | `SIB` | Classifier cache directory to analyze. |
| `--mode {xmin,sample}` | required | `xmin` sweeps the lower cutoff; `sample` sweeps sample size at one cutoff. |
| `--n_synth N` | `500` | Monte Carlo synthetic datasets per goodness-of-fit estimate; use `2500` for publication runs. |
| `--n_xmin N` | `30` | Number of cutoff-grid points in `xmin` mode. |
| `--xmin FLOAT` | auto-detected | Fixed lower cutoff in seconds for `sample` mode. |
| `--step N` | `1` | Load only `seq_<index>.npz` files whose molecule index is divisible by N. |
| `--seed N` | `42` | Monte Carlo random seed. |

### `distr_pnm_connectivity`

Installed command: `gas-traj-pnm-connectivity-distr`.

| Parameter | Required/default | Meaning |
|---|---|---|
| `--pnm PATH:LABEL` | required, repeatable | PNM directory and plot-series label. Repeat to compare datasets in the interactive plot. |

### `distr_pnm_count_pores_throats`

Installed command: `gas-traj-pnm-pore-throat-distr`.

| Parameter | Required/default | Meaning |
|---|---|---|
| `--pnm PATH:LABEL` | required, repeatable | PNM directory and series label. Repeat to compare pore/throat count histograms. |

### `msdt_builder`

Installed command: `gas-traj-msdt-builder`.

| Parameter | Required/default | Meaning |
|---|---|---|
| `--trj PATH:LABEL:STEP` | required, repeatable | Data-directory prefix, plot label, and positive molecule stride. The current implementation appends `trj.gro` and `msd_time_avarage.csv` directly, so `PATH` must end with `/`. Repeat to overlay datasets. |

### `inv_plot`

| Parameter | Required/default | Meaning |
|---|---|---|
| `trajectory` | required | Input multi-molecule GRO trajectory. |
| `--trajectory-index N`, `--num N` | `0` | Zero-based molecule/trajectory index. The two names are aliases. |
| `--mu FLOAT` | required | DM distance scale; one of `0.5`, `1.0`, `1.5`, `2.0`, `2.5`, or `3.0`. |
| `--nu FLOAT` | `0.1` | Invariant threshold above which points are marked trapped. |
| `--diag-percentile {0,5,10,50}` | `5` | Reference percentile controlling how many distance-matrix diagonals are filled. |
| `--kernel-size N` | `3` | Radius of the DM smoothing kernel; actual width is `2N + 1`. |
| `--p-value FLOAT` | `0.9` | Quantile used to obtain the critical minimum trap-run length. |
| `--traj-type {fBm,Bm}` | `fBm` | Reference-motion threshold table. |
| `--x-axis {index,time-ps,time-us}` | `index` | Horizontal-axis units. |
| `--min-index N` | `0` | First included trajectory point. |
| `--max-index N` | last point | Last included point, inclusive. |
| `--filter-short-traps` | off | Apply critical-length filtering to the shaded trapping regions. |
| `--trap-fill-y-max FLOAT` | `nu` | Upper Y coordinate of trap shading. |
| `--output FILE` | generated SVG below the trajectory’s `figs/` directory | Explicit output figure path. |

## Simulation and interactive visualization

### `atom_visualization`

Installed command: `gas-traj-atom-legend`. It has no parameters and displays
a fixed five-atom color legend made from synthetic coordinates.

### `simulate_trajectory`

Installed command: `gas-traj-simulate-trajectory`.

| Parameter | Required/default | Meaning |
|---|---|---|
| `path` | required | Data directory containing `radiuses.npy` and `throat_lengths_weibull_fitter.json`. |
| `--k FLOAT` | `0.5` | Probability of taking an intra-trap dwell segment after a move instead of immediately continuing between traps. |
| `--p FLOAT` | `0.5` | Probability of returning rather than exploring a new trap. |
| `--steps N` | `1000` | Number of simulated trajectory steps. |
| `--radius FLOAT` | `0.02` nm | Rendered trajectory tube/point radius; it does not affect simulation. |

### `simulate_pnm_trajectory`

Installed command: `gas-traj-simulate-pnm-trajectory`.

| Parameter | Required/default | Meaning |
|---|---|---|
| `pnm_prefix` | required | Common PNM path prefix without `_node1.dat`, `_node2.dat`, or `_link1.dat`. |
| `--k FLOAT` | `0.5` | Probability of taking an intra-trap dwell segment after a move instead of immediately continuing between pores. |
| `--p FLOAT` | `0.5` | Probability of returning along walk history rather than exploring a neighbor. |
| `--steps N` | `1000` | Number of simulated steps. |
| `--radius FLOAT` | `0.02` nm | Render-only trajectory tube/point radius. |
| `--min-radius FLOAT` | `0.0` nm | Remove pores with radius less than or equal to this threshold before walking. |

### `vis_atoms_struct`

| Parameter | Required/default | Meaning |
|---|---|---|
| `path` | required | Data directory. The current script expects `type1.ch4.300.gro`, `ker.pdb`, and optionally cached `structures/` below it. |
| `img` | required | Distance-field `.npy` image whose filename contains bounding-box metadata. |
| `--index N` | required | Simulation step (`num=`) of the structure frame. |
| `--time-ps N` | required | Frame time in ps, used in the structure-cache filename. |

### `vis_slice_struct`

| Parameter | Required/default | Meaning |
|---|---|---|
| `input` | required | Binary or float structure-image `.npy` file with standard filename metadata. |
| `output` | required | Output 2D figure path. |
| `--size FLOAT` | `0.4` nm | Physical edge length of the leading cube from which the central X slice is taken. |
| `--ref-size FLOAT`, `--ref_size FLOAT` | unset | Optional physical reference length in nm; reports its corresponding pixel count for diagnostics. The spellings are aliases. |

### `vis_struct_example`

Installed command: `gas-traj-vis-struct-example`.

| Parameter | Required/default | Meaning |
|---|---|---|
| `--float_image_path FILE` | required | Distance-field `.npy` produced by `distance_map_structs`; its filename supplies the bounding box. |
| `--isovalue FLOAT` | `0.11` | Isosurface threshold applied to the distance field. |
| `--img-opacity FLOAT` | `0.5` | Rendered surface opacity. |

### `vis_struct_pnm`

| Parameter | Required/default | Meaning |
|---|---|---|
| `img` | required | Distance-field `.npy` image. |
| `pnm_prefix` | required | Common PNM files prefix without the Statoil filename suffixes. |

### `vis_struct_trajectory`

| Parameter | Required/default | Meaning |
|---|---|---|
| `path` | required | Data directory containing `trj.gro`. |
| `img` | required | Distance-field `.npy` image with bounding-box metadata in its filename. |
| `--num N` | `2` | Zero-based molecule index. |
| `--start-index N` | `0` | First trajectory point retained for the overlay. |
| `--end-index N` | `-1` | Exclusive crop endpoint; a value less than or equal to zero means the trajectory end. |

### `vis_traject`

| Parameter | Required/default | Meaning |
|---|---|---|
| `trj` | required | Input multi-molecule GRO trajectory. |
| `num` | required | Zero-based molecule index. |
| `--traps FILE` | unset | Optional `traps_<index>.npz`; enables trap/free transition coloring instead of distance coloring. |

## External PyMOL and VMD renderers

PyMOL parameters are normally shlex-quoted in `PYMOL_SCRIPT_ARGS`; PyMOL
scripts also accept arguments after a literal `--` when the host invocation
preserves them. VMD scripts first read arguments passed after `-args` and fall
back to the whitespace-split `VMD_SCRIPT_ARGS`. Use direct `-args` for an
option such as `--cell` whose single value contains spaces. Renderers display
interactively when no output path is supplied.

### `visualize_atom_legend.pml`

| Parameter | Required/default | Meaning |
|---|---|---|
| `--output FILE` | unset | PNG path for the labeled atom legend. |
| `--output-spheres FILE` | derived/unset | PNG path for the spheres-only legend. With `--output`, defaults to `<output-stem>_spheres_only.png`. |
| `--width N` | `2200` px | PNG width. |
| `--height N` | `650` px | PNG height. |
| `--dpi N` | `300` | PNG resolution metadata. |
| `--ray` / `--no-ray` | `--no-ray` | Enable PyMOL ray tracing or use the OpenGL framebuffer. |

### `visualize_kerogen_molecule.pml`

| Parameter | Required/default | Meaning |
|---|---|---|
| `--ker-pdb FILE` | `ker.pdb` | PDB containing kerogen topology. |
| `--chain ID` | `A` | PDB chain selected for rendering. |
| `--resi ID`, `--resid ID` | `1` | Residue number; the names are aliases. |
| `--output FILE` | unset | Optional PNG destination. |
| `--width N` | `2200` px | PNG width. |
| `--height N` | `1800` px | PNG height. |
| `--dpi N` | `300` | PNG resolution metadata. |
| `--ray` / `--no-ray` | `--no-ray` | Choose ray tracing or framebuffer output. |
| `--camera-rot X Y Z` | `90 0 0` degrees | Camera rotations after orienting the molecule. |
| `--zoom-buffer FLOAT` | `5.0` Å | Extra space passed to PyMOL zoom. |

### `visualize_kerogen_part_cell.pml`

| Parameter | Required/default | Meaning |
|---|---|---|
| `--ker-pdb FILE` | `ker.pdb` | PDB topology/cell source. |
| `--sim-gro FILE` | unset | Optional GRO trajectory supplying coordinates for the selected frame. |
| `--frame N` | `0` | Zero-based GRO frame, ignored when `--step` or `--time` is supplied. |
| `--step N` | unset | Select a frame by the `step=` value in its GRO title; takes precedence over `--time` and `--frame`. |
| `--time FLOAT` | unset | Select a frame by GRO-title time in ps; takes precedence over `--frame`. |
| `--resname NAME` | `KRG` | Residue name read from GRO; `ALL` disables residue-name filtering. |
| `--box-size FLOAT` | `20.0` Å | Side of a center-defined cubic subbox. |
| `--center X Y Z` | PDB cell center | Center of the cubic subbox in Å. |
| `--xmin/--xmax/--ymin/--ymax/--zmin/--zmax FLOAT` | unset | Six explicit subbox boundaries in Å. Supply the complete set to override `--center`/`--box-size`. |
| `--output FILE` | unset | Optional PNG destination. |
| `--width N` | `2200` px | PNG width. |
| `--height N` | `1800` px | PNG height. |
| `--dpi N` | `300` | PNG resolution metadata. |
| `--ray` / `--no-ray` | `--no-ray` | Choose ray tracing or framebuffer output. |
| `--surface-mode {all,none}` | `all` | Show or suppress molecule surfaces. |
| `--surface-transparency FLOAT` | `0.4` | PyMOL surface transparency. |
| `--camera-rot X Y Z` | `0 0 0` degrees | Camera rotations after orienting the subbox. |
| `--zoom-buffer FLOAT` | `3.0` Å | Extra space around the selected subbox. |

### `visualize_kerogen_part_cell_with_molecule.pml`

| Parameter | Required/default | Meaning |
|---|---|---|
| `--ker-pdb FILE` | required | PDB providing bonds/topology. |
| `--sim-gro FILE` | required | GRO trajectory providing coordinates. |
| `--frame N`, `--step N` | `0` | Zero-based GRO frame; these names are aliases in this script (not a simulation-step lookup). |
| `--mol-index ID`, `--mol-id ID` | required | Residue/molecule number highlighted in the selected subbox. |
| `--box-size FLOAT` | `20.0` Å | Side of the cubic subbox. |
| `--resname NAME` | `KRG` | GRO residue-name filter; `ALL` accepts every residue. |
| `--center X Y Z` | simulation-cell center | Cubic subbox center in Å. |
| `--output FILE` | unset | Optional PNG destination. |
| `--width N` | `2200` px | PNG width. |
| `--height N` | `1800` px | PNG height. |
| `--dpi N` | `300` | PNG resolution metadata. |
| `--camera-rot X Y Z` | `0 110 0` degrees | Fixed camera rotations after orienting the subbox. |
| `--surface-mode {selected,all,none}` | `all` | Render a surface for only the highlighted molecule, for all molecules, or for none. |
| `--ray` / `--no-ray` | `--ray` | Choose ray tracing or framebuffer output. |

### `visualize_kerogen_cell.tcl`

| Parameter | Required/default | Meaning |
|---|---|---|
| `--ker-pdb FILE` | `ker.pdb` | PDB cell loaded by VMD. |
| `--cell "A B C ALPHA BETA GAMMA"` | `62.309 74.106 130.150 90 90 90` | Periodic-cell lengths in Å and angles in degrees. Pass the quoted six-value argument through direct VMD `-args`, not the whitespace-split environment fallback. |
| `--periodic-axes TEXT` | `xyz` | Axes for displayed periodic replicas; use any combination of `x`, `y`, and `z`. |
| `--periodic-count N` | `1` | Number of periodic replicas shown by VMD. |
| `--rotate-x FLOAT` | `0` degrees | Final X rotation. |
| `--rotate-y FLOAT` | `110` degrees | Final Y rotation. |
| `--rotate-z FLOAT` | `0` degrees | Final Z rotation. |
| `--scale FLOAT` | `1.6` | Final VMD view scale multiplier. |
| `--output FILE` | unset | Optional VMD snapshot destination. |

### `visualize_kerogen_part_cell.tcl`

| Parameter | Required/default | Meaning |
|---|---|---|
| `--ker-pdb FILE` | `ker.pdb` | PDB cell loaded by VMD. |
| `--box-size FLOAT` | `30.0` Å | Side of a center-defined cubic subbox. |
| `--center X Y Z` | PDB cell center | Center of the cubic subbox in Å. |
| `--xmin/--xmax/--ymin/--ymax/--zmin/--zmax FLOAT` | unset | Complete explicit subbox boundaries in Å; when all six are present they override center/size. |
| `--subbox-pdb FILE` | `subbox.pdb` | Temporary/extracted PDB written for the selected atoms. |
| `--rotate-x FLOAT` | `0` degrees | Final X rotation. |
| `--rotate-y FLOAT` | `0` degrees | Final Y rotation. |
| `--rotate-z FLOAT` | `0` degrees | Final Z rotation. |
| `--scale FLOAT` | `1.0` | Final VMD view scale multiplier. |
| `--output FILE` | unset | Optional VMD snapshot destination. |
