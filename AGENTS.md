# AGENTS.md

Guidance for AI agents (and human contributors) adding to or modifying QuantUS-Plugins. See [docs/architecture.md](docs/architecture.md) first for how the six plugin stages and core data objects fit together.

## Adding a new plugin

Every one of the six stages (image loading, segmentation loading, analysis config loading, analysis, visualizations, data export) follows the same shape. See [docs/templates/new_plugin_checklist.md](docs/templates/new_plugin_checklist.md) for a quick per-stage reference table.

1. **Read the stage's README first** — `quantus/<stage>/README.md` documents the exact function/class signature and decorators that stage expects. Don't guess the contract from a neighboring stage; each has small but important differences (e.g. `seg_loading` plugins are wrapped in a dict by the `extensions` decorator, unlike every other stage's plain-attribute decorators).
2. **Look at an existing plugin in that stage as a template** — e.g. `quantus/image_loading/utc_loaders/clarius_rf/` for image loading, `quantus/analysis/paramap/analysis_methods/attenuation_coef.py` for an analysis method.
3. **Drop your plugin in the right place, matching the existing naming convention** (a new folder for folder-based stages, a new file for single-file stages). Discovery is automatic — see `quantus/plugin_utils.py`'s `discover_plugin_classes`/`discover_marked_functions`, used by every stage's `options.py`. **Never hardcode a plugin list anywhere** — always go through the relevant `get_*` function in that stage's `options.py`.
4. **Verify discovery picked it up**: run `pytest` (see Testing below) or directly call the relevant `get_*` function and confirm your plugin appears in the returned dict with the attributes your decorators set.

## Private / external plugins

Plugins don't have to live in this repo. `quantus/plugin_utils.py`'s `get_external_plugin_root()` scans `QUANTUS_PLUGIN_DIR` (default `.quantus/plugins/` at the repo root), mirroring the in-repo layout one level down (e.g. `$QUANTUS_PLUGIN_DIR/analysis/paramap/analysis_methods/your_method.py`). Anything found there is merged in **additively** alongside the built-in public plugins — it never replaces or hides them, except that a same-named external plugin overrides a built-in one (with a `warnings.warn` at discovery time).

This is the intended distribution mechanism for private/proprietary plugins: a plugin owner emails a single plugin folder/file to an approved user, who drops it into their own `QUANTUS_PLUGIN_DIR` — nothing proprietary ever needs to touch this repo's git history. It's also the mechanism a future native GUI would use to load plugins dynamically at runtime.

A folder-based external plugin (e.g. a new image loader) can use ordinary relative imports **to its own sibling files** (`from .parser import X`), same as a built-in one — `import_external_module` in `plugin_utils.py` handles this by putting the plugin's directory on `sys.path` before importing. Plugin/file names only need to be unique within the directory being scanned (the same guarantee the filesystem already gives built-in plugins), not globally across all six stages.

**Relative imports reaching further than the plugin's own folder will break once it's external** — e.g. a built-in image loader can do `from ....data_objs.image import UltrasoundRfImage` or `from ..transforms import scanConvert` because it's nested 3-4 levels inside the real `quantus` package; once that same file is dropped into `QUANTUS_PLUGIN_DIR`, it's only ever nested one level deep (under its own plugin folder), so those dotted imports resolve to nothing and raise `ImportError: attempted relative import beyond top-level package`. **A plugin destined for `QUANTUS_PLUGIN_DIR` must use absolute imports for anything outside its own folder** — e.g. `from quantus.data_objs.image import UltrasoundRfImage`, `from quantus.image_loading.utc_loaders.transforms import scanConvert` — and this applies even to *built-in* plugin files if you're moving them out to `QUANTUS_PLUGIN_DIR` (fix the imports as part of the move, don't just relocate the folder as-is).

If you're asked to work on a private plugin, check whether it should actually go under `QUANTUS_PLUGIN_DIR` rather than into this repo — this repo is public, and proprietary plugins committed here defeat the point of the mechanism above.

## Testing

Run `pytest` from the repo root (or let `.github/workflows/tests.yml` run it in CI). The existing suite (`tests/test_plugin_discovery.py`, `tests/test_external_plugin_dir.py`) is intentionally scoped to **plugin-discovery smoke tests** — every stage's `get_*` function returns at least one plugin, and every discovered plugin carries the attributes the pipeline relies on (see each stage's README for what those are) — plus coverage of the external-plugin-dir merge/collision behavior. There's no end-to-end pipeline test with synthetic scan data yet.

When adding a new stage-level `get_*` function or changing the discovery contract in `plugin_utils.py`, extend `tests/test_plugin_discovery.py` accordingly. When adding a new plugin, you don't need to write a dedicated test for it — the existing smoke tests already assert every discovered plugin (including yours) satisfies its stage's contract.

## Code reuse

- Plugin discovery and decorator boilerplate belongs in `quantus/plugin_utils.py` — don't reintroduce a bespoke folder-scanning loop or a one-off `def decorator(func): func.x = args; return func` in a new stage; use `discover_plugin_classes`, `discover_marked_functions`, or `attr_decorator` instead.
- Each stage's `transforms.py` holds shared helpers usable across multiple plugins in that stage — check there before duplicating logic across plugin files in the same stage.
