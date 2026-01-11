# Repository Guidelines

## Project Structure & Module Organization
- `python/sglang/`: core Python runtime, model code, and server launchers.
- `sgl-kernel/`: separate kernel package maintained alongside the main runtime.
- `sgl-model-gateway/`: gateway services (Rust and Python).
- `test/`: tests (`test/srt` backend runtime, `test/lang` frontend language, `test/srt/nightly` nightly suites).
- `docs/`: documentation and notebooks; see `docs/README.md` for the docs workflow.
- `benchmark/`, `examples/`, `scripts/`, `docker/`, `assets/`: supporting tooling, samples, and infrastructure.

## Build, Test, and Development Commands
- `pip install -e "python"`: editable install for local development.
- `python3 -m sglang.launch_server --model <hf_model>`: run a local server for manual testing.
- `pre-commit run --all-files`: run linting/format checks before opening a PR.
- `cd test/srt && python3 test_srt_endpoint.py`: run a backend unit test file.
- `cd test/srt && python3 run_suite.py --suite per-commit`: run a CI suite locally.
- `make compile && make html` (from `docs/`): build documentation; `bash serve.sh` for live preview.

## Coding Style & Naming Conventions
- Indentation: 4 spaces by default; 2 spaces for JSON/YAML/Markdown; Makefiles use tabs (`.editorconfig`).
- Formatting and linting are enforced via pre-commit: `isort`, `ruff` (F401/F821), `black-jupyter`, `clang-format`, `codespell`, `nbstripout`.
- Keep runtime code efficient and modular: avoid duplication, minimize CPU-GPU sync, keep files under 2,000 LOC, prefer pure functions.

## Testing Guidelines
- Primary framework: Python `unittest`; if using pytest, include `pytest.main([__file__])`.
- Add tests under `test/srt` or `test/lang`; nightly-only tests go in `test/srt/nightly/`.
- Register suites in `test/srt/run_suite.py` (or `test/lang/run_suite.py`) and keep suite lists alphabetical.

## Commit & Pull Request Guidelines
- Commit messages are short and imperative; optional scope tags like `[bug fix]`, `[diffusion]`, or `fix:` are common, and merged PRs include `(#NNNNN)`.
- Use a feature branch (no direct commits to `main`).
- Follow `.github/pull_request_template.md`: include Motivation/Modifications, accuracy tests for output changes, benchmarks for performance changes, and ensure pre-commit/tests/docs items are addressed.
- CI runs on PRs with the `run-ci` label; authorized users can use `/tag-run-ci-label` or `/rerun-failed-ci`.
