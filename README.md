# Lang2Drive

Code and artifact release for the Lang2Drive paper. This repository contains the public-safe scene-generation pipeline, evaluation utilities, a sanitized workbook template, and a compact set of curated prompt/code examples.

## Overview

Lang2Drive provides a workbook-driven workflow for preparing CARLA scenarios, running generated code, and evaluating outputs. The public release keeps the reproducible pipeline and representative prompt/code artifacts while excluding large rendered scene dumps and private runtime history.

## Repository Contents

- `agent_skill_scene_loop.py`: scene preparation, ready checks, and workbook utilities
- `agentic_wine_handoff_runner.py`: manifest-driven execution entrypoint
- `agentic_visual_evaluator.py`: evaluation utilities
- `agentic_wine_time_weather_matrix_runner.py`: shared time x weather matrix runner
- `agentic_scene_generator.py`: orchestrates workbook-driven scene generation
- `carla_wine_bridge.py`: runtime adapter for local and Wine-backed execution
- `scene_excel_utils.py`, `research_scene_history_excel.py`, `shot_history_excel.py`: workbook helpers
- `scene_utils/`: shared time/weather support code
- `keyframe_selection.py`: CLIP-based keyframe selection utility restored from an earlier paper-stage commit
- `run_preannotation.py`, `run_gpt.sh`, `prompts.json`, `test.py`: GPT-based preannotation utilities restored from earlier paper-stage commits
- `generated_prompts/`, `generated_code/`: five curated prompt/code examples used as public artifacts
- `tests/`: non-CARLA unit tests for the released workflow

See `ARTIFACT_INDEX.md` for the exact curated prompt/code folders shipped with this release.

## Setup

Requirements:

- Python 3.9+
- A separately installed CARLA runtime
- On macOS, `CARLA_APP_PATH` should point to the local `CARLA.app` bundle when using the Wine bridge

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

The keyframe-selection utility also relies on `open-clip-torch` and its `torch` backend, which are included in `requirements.txt`.

## Reproducing the Pipeline

Inspect the sanitized workbook template:

```bash
python3 agent_skill_scene_loop.py list-scenes --excel "Keyword Prompt Verification.xlsx"
```

Prepare the next scene handoff:

```bash
./01_generate_scene.sh
```

Mark generated code ready:

```bash
python3 agent_skill_scene_loop.py mark-ready --handoff-dir handoffs
```

Run the latest prepared scene:

```bash
python3 agentic_wine_handoff_runner.py --handoff-dir handoffs
```

Evaluate the latest output:

```bash
./03_evaluate_latest.sh
```

Run the shared matrix benchmark:

```bash
python3 agentic_wine_time_weather_matrix_runner.py --handoff-dir handoffs
```

Optional annotation utilities:

```bash
python3 keyframe_selection.py --input_root /path/to/frames --output_root /path/to/keyframes
./run_gpt.sh --input_dir /path/to/images --output_dir /path/to/preannotations
```

## Release Scope

Included in this repository:

- The maintained public workflow and helper modules
- A sanitized workbook template
- Five curated prompt/code examples
- Keyframe-selection and GPT preannotation utilities from earlier paper-stage commits
- Public documentation and non-CARLA tests

Intentionally excluded:

- Rendered `scenes/` outputs and other large binary result folders
- Historical handoffs, intervention reports, and private run logs
- Legacy/internal planning material
- Third-party research bundles and copied reference repos

## Configuration

- `CARLA_APP_PATH`: optional macOS path to the CARLA bundle used by the Wine bridge
- `CARLA_WINE_PYTHON_EXE`: optional Windows Python path inside the Wine prefix
- `SCENE_GENERATOR_CODEX_MODEL`: optional model override for evaluator CLI integrations

On non-macOS platforms, the runtime defaults to `local`. On macOS, `auto` resolves to `wine-bridge`.

## Verification

Run the non-CARLA test suite:

```bash
python3 -m unittest discover -s tests -q
```
