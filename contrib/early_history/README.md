# Early History Contributor Utilities

This directory keeps a small set of utilities recovered from the private repository history so collaborator-era work is represented in the public release.

Included here:

- `camera_sim/`: the original camera capture script and launcher from the earliest CARLA prototype
- `scene_edit/`: the editable simulation utility and the later scene-edit RAG helpers

These files are intentionally separated from the main Lang2Drive workflow:

- they are not part of the supported paper reproduction path
- they retain their historical structure where useful
- they may require extra dependencies or CARLA setup beyond the main public workflow

History notes:

- `camera_sim/` traces back to commit `d5a4fa9`
- `scene_edit/carla_editable_sim_w_latency.py` traces back to commit `3b665ba`
- `scene_edit/rag_scenerio_generator/` and `scene_edit/carla_scenarios_db.json` trace back to the `scene-edit` branch history, including commit `68952cd`

The Git metadata available in this clone does not preserve a distinct `Zarif` author identity, so these utilities are grouped by source history rather than attributed to a specific person in this public repo.
