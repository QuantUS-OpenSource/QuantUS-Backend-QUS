# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

QuantUS-Plugins is a quantitative ultrasound (QUS) analysis framework built on an extensible plugin architecture. It loads RF ultrasound data from various manufacturers, applies configurable QUS analysis methods over a segmented region using a sliding-window technique, and exports parametric maps, visualizations, and numerical features. It's used both as a library (via the GUI in the separate QuantUS repo) and standalone via CLI/YAML config/scripting.

See [docs/architecture.md](docs/architecture.md) for the full plugin-discovery pattern and core data objects, and [AGENTS.md](AGENTS.md) for the end-to-end workflow for adding a new plugin.

## Setup and running

Requires Python 3.10.

```bash
python3.10 -m virtualenv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install pyradiomics==3.0.1 --no-build-isolation
```

There is no `pyproject.toml`/console-script install — the package is run directly out of the repo checkout. Entry points, both wired to `core_pipeline` in `quantus/full_workflow.py`:
- `python -m quantus.full_workflow $CONFIGPATH` — runs the full workflow from a YAML config (see `configs/sample.yaml` for the schema: scan/seg/config loaders + kwargs, analysis type + funcs, visualization type + funcs, data export type + funcs).
- `main_cli()` in `quantus/full_workflow.py` — argparse-based (JSON-string kwargs) equivalent of the above, callable directly from Python; not currently wired to a console script.
- Scripting: `quantus/entrypoints.py` exposes each pipeline stage (`scan_loading_step`, `seg_loading_step`, `analysis_config_step`, `analysis_step`, `visualization_step`, `data_export_step`) as an individually callable function, used for custom/batch pipelines (see `quantus/processing/` and `CLI-Demos/*.ipynb` for examples).

Plugin-discovery smoke tests exist under `tests/` (run with `pytest`) and CI runs them on push/PR — see `.github/workflows/tests.yml`. There's no broader test suite or linter config beyond that; verify other changes by running the CLI/YAML workflow against a sample scan or by exercising the relevant `entrypoints.py` step directly.
