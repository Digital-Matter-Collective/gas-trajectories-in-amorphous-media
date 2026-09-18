# Security Policy

## Supported versions

This is a research codebase with a single actively maintained line: the
latest commit on `main` (and the most recent tagged release, currently
`v1.0.0`). There is no long-term support for older tags; fixes land on
`main` and are not backported.

## Reporting a vulnerability

Please **do not** open a public GitHub issue for a security vulnerability.
Instead, use GitHub's private vulnerability reporting for this repository:

<https://github.com/Digital-Matter-Collective/gas-trajectories-in-amorphous-media/security/advisories/new>

This opens a draft security advisory visible only to the maintainers, where
you can describe the issue, its impact, and (if you have one) a suggested
fix. We'll acknowledge reports and follow up as we investigate; there is no
bug bounty.

## Scope and threat model

This package processes molecular-dynamics trajectory files and pore-network
model (PNM) files supplied by the user running it, and writes cached
intermediate results and figures back to a user-specified directory. It is
research software run locally or on a compute cluster on data the user
already controls, not a network-facing service — the realistic threat model
is a malicious or corrupted **input file** (trajectory, PNM, or cached
result) being loaded into a process running with the user's own privileges,
not a remote attacker.

### Deserialization of untrusted data

`pickle.load` can execute arbitrary code from a maliciously crafted payload,
so it must never be used with a path that a user (or a downstream script) can
point to untrusted data. Since the migration away from `pickle`, every
first-party cache, checkpoint, and intermediate result in this repository —
trap sequences, fitted distribution parameters, extracted MD structures, and
benchmark checkpoints — is written and read as `.npz`/`.npy` (NumPy array
formats) or JSON. NumPy files are loaded with `allow_pickle=False` (the
default), so these formats do not deserialize embedded Python objects.

The one remaining use of `pickle.load` in the codebase
(`processes/trajectory_analyzer/dm.py`, loading files under
`list_threshold/`) is intentionally out of scope for that migration: those
24 `.pkl` files are static lookup tables bundled *inside the package itself*
(declared in `pyproject.toml`'s `[tool.setuptools.package-data]`) and are
never read from a user- or CLI-supplied path — they ship with the code, not
with any dataset a user points the tools at.

If you find a code path where a `.pkl` or `.pickle` payload from a
user-controlled location (a CLI argument, a data directory, a downloaded
archive) is deserialized, please report it as above — that would be exactly
the class of issue this policy exists to prevent.

### Dependencies

Supported dependency ranges are declared in `pyproject.toml`; the exact
reference environment is recorded in `environment.yml`. If you find a
vulnerability in a dependency that materially affects this project (not just
a theoretical CVE in an unused code path), a report is still welcome so we
can evaluate whether to update the affected version constraint.
