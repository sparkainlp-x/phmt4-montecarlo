# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/). All numbers are SYNTHETIC.

## [0.2.0] - 2026-09-27

### Changed
- v2 density-matrix redesign (`phmt4/v2.py`): exact unitary step with Lindblad dephasing (Strang splitting) replaces the first-order non-unitary step, which acted as power iteration; the crystallization mirror now acts on mixed states instead of being a no-op on pure states.
- Mirror-off ablation (`--no-mirror`), parameter sweep (`scripts/sweep.py`) and numerical checks (`scripts/v2_checks.py`).
- 18 tests; CI on Python 3.11 and 3.12.
- `.zenodo.json` metadata and CHANGELOG.

## [0.1.0] - tag `v1-original`

### Added
- Faithful original prototype with minimal fixes, preserved for comparison.
