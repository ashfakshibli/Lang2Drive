# Scene Generation Reference

## 1. Goal

Use this repository to prepare CARLA scene prompts, generate runnable scene scripts, execute them through either a local runtime or a Wine bridge, and evaluate the resulting frames.

## 2. Supported Entry Points

- `agent_skill_scene_loop.py`: prepare scenes from the workbook and validate ready-to-run code
- `agentic_scene_generator.py`: direct workbook-driven generation path
- `agentic_wine_handoff_runner.py`: execute a manifest-selected scene
- `agentic_visual_evaluator.py`: evaluate rendered frames
- `agentic_wine_time_weather_matrix_runner.py`: run the shared time x weather matrix

## 3. Workbook Contract

The workbook parser expects these headers by name:

- `Keyword`
- `Prompt`
- `Scene Specifications`

Optional headers:

- `Code Output`
- `Verification`

The checked-in workbook is a sanitized template with five example scenes.

## 4. Runtime Modes

- `local`: run the scene script directly with the host Python interpreter
- `wine-bridge`: run through the CARLA Wine bundle on macOS
- `auto`: resolve to `wine-bridge` on macOS and `local` elsewhere

Relevant environment variables:

- `CARLA_APP_PATH`
- `CARLA_WINE_PYTHON_EXE`
- `SCENE_GENERATOR_CODEX_MODEL`

## 5. Output Conventions

Stage 1 prepares:

- `generated_prompts/<scenario>/<run_id>/shot_<n>/`
- `generated_code/<scenario>/<run_id>/shot_<n>/`
- `handoffs/<run_id>/<scenario>/manifest.json`

Stage 2 writes scene outputs under:

- `scenes/<scenario>/<run_id>/shot_<n>/`

Stage 3 writes evaluation metadata back to the handoff directory.

## 6. Evaluation Workflow

- Use `--manual-only --strict-intent` for manual evaluation mode
- Keep frame coverage, key frames, and intent notes together with the selected manifest
- Use the latest-manifest pointer files in `handoffs/` unless you need an explicit manifest path

## 7. Matrix Runs

The matrix runner reads `scene_utils/time_weather_matrix_spec.json` and executes the shared 20-variant time x weather sweep for a finalized scene manifest.

The retained `scenes/red_light_violation/20260220_122357/shot_0/` sample includes:

- `standalone_noon_clear/`
- `time_weather_matrix/`

## 8. Primary External References

- CARLA Python API: <https://carla.readthedocs.io/en/latest/python_api/>
- Synchrony and timestep: <https://carla.readthedocs.io/en/latest/adv_synchrony_timestep/>
- Traffic Manager: <https://carla.readthedocs.io/en/latest/adv_traffic_manager/>
- Actors and blueprints: <https://carla.readthedocs.io/en/latest/core_actors/>
- Sensors: <https://carla.readthedocs.io/en/latest/core_sensors/>

## 9. Static Mesh Guidance

When a dedicated CARLA blueprint does not exist, prefer `static.prop.mesh` over ad hoc stand-ins.

Basic pattern:

```python
bp = bp_lib.find("static.prop.mesh")
bp.set_attribute("mesh_path", "/Game/Carla/Static/Vegetation/Trees/SM_BeechTree_3.SM_BeechTree_3")
bp.set_attribute("mass", "0")
actor = world.try_spawn_actor(bp, transform)
```

Useful categories:

- Trees: `/Game/Carla/Static/Vegetation/Trees/...`
- Bushes: `/Game/Carla/Static/Vegetation/Bushes/...`
- Rocks: `/Game/Carla/Static/Vegetation/Rocks/...`
- Traffic assets: `/Game/Carla/Static/TrafficSign/...`

Use physics-enabled mass values only when the scene requires dynamic motion or fall behavior.
