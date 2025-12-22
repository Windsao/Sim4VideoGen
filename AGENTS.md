# Repository Guidelines

## Project Structure & Module Organization
DiffSynth-Studio installs as a Python package via `setup.py`, and almost all core logic lives in `diffsynth/` (pipelines, models, trainers, schedulers, VRAM management). Model recipes are grouped in `examples/`—each folder (such as `examples/qwen_image`, `examples/flux`, and `examples/wanvideo`) provides paired `model_inference/` and `model_training/` scripts you can copy when adding a new variant. Web tooling and Ops helpers sit inside `apps/gradio` and `apps/streamlit`. High-level launchers in the repo root (`inference_wan.py`, `run_multigpu_inference.sh`, `train_wan22_stage*.sh`) show how to wire datasets, prompts, and accelerators; keep new workflows beside these scripts. Data artifacts (metadata CSVs, prompt JSON, validation videos) are parked under `data/`, `phygenbench_results/`, and `wandb/`, so reference them rather than duplicating assets.

## Build, Test, and Development Commands
- `pip install -e .` — installs the package in editable mode and pulls in `requirements.txt`.
- `python inference_wan.py` — runs the default Wan text-to-video demo and validates GPU, tokenizer, and VRAM-management plumbing.
- `bash run_multigpu_inference.sh` — spawns four coordinated processes using `inference_physics_iq_wan22_multigpu.py`; edit `WORLD_SIZE` and `MODEL_NAME` when scaling beyond 4 GPUs.
- `python -m pytest test_model_paths.py -- \"[\\\"/path/to/weights\\\"]\"` — confirms checkpoint paths resolve before launching a job.
- `python test_data_loading.py` — verify dataset metadata/ops; adjust the hard-coded paths before committing.

## Coding Style & Naming Conventions
- Python only: 4-space indentation, strict `snake_case` for functions/variables, `CamelCase` for classes; mirror the style in `diffsynth/pipelines/flux_image_new.py`.
- Prefer explicit imports, type hints, and `dataclass` definitions for structs (see `ControlNetInput`) so schedulers and runners remain inspectable.
- Keep configuration in `ModelConfig` blocks or YAML under `diffsynth/configs/`; keep filenames descriptive (`train_wan_with_motion.py`, not `train_new.py`).
- Document non-obvious tensor tricks or distributed barriers inline; short `#` comments before complex sections are preferred over long docstrings.

## Testing Guidelines
Use `pytest` to run any `test_*.py` file so failures show up in CI. GPU-heavy or path-dependent tests (data loading, multi-GPU inference) must guard against missing assets—add arguments or environment switches instead of hard-coded cluster paths. When adding a pipeline, include a smoke test that executes a single forward pass with a `--max_samples` flag similar to `run_multigpu_inference.sh`, and record expected artifacts under `test_outputs_*`. Report coverage by showing which scripts were exercised in the PR description.

## Commit & Pull Request Guidelines
History favors short, imperative subjects (`fix data bug`, `add warp loss`). Keep commits focused on one change-set and mention affected scripts or models in the body. Every PR should include: (1) a summary of the scenario (model, dataset, resolution), (2) commands used (`pip install -e .`, scripts, `torchrun` flags), (3) qualitative artifacts or metrics (attach frames or point to `test_outputs_*`), and (4) any dependency bumps. Link related issues, update README/example files when behavior changes, and request model owners for review when touching their subdirectories.

## Security & Configuration Tips
Model downloads rely on `modelscope.snapshot_download` and optional `local_model_path` overrides; keep credentials in your ModelScope CLI config or environment variables—never in tracked files. Large checkpoints, datasets, and W&B runs belong in paths already ignored (`models/`, `data/`, `wandb/`); add new directories to `.gitignore` before syncing generated assets. When contributing scripts, accept `--model_root`, `--dataset_root`, or `LOCAL_*` env vars so other agents can reuse the code without mounting `/nyx-storage1`. Sanitise prompts and sample outputs before sharing in issues or documentation.
