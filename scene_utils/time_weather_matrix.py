from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union


DEFAULT_SPEC_PATH = Path(__file__).with_name("time_weather_matrix_spec.json")
SCENE_DEFAULT_KEY = "scene_default"
VALID_STREETLIGHT_MODES = {"auto", "on", "off"}
_TIME_ALIASES = {
    "almost_night": "almost_night_no_streetlights",
    "almost_night_no_light": "almost_night_no_streetlights",
    "almost_night_no_lights": "almost_night_no_streetlights",
}


class TimeWeatherSpecError(ValueError):
    """Raised when the matrix spec is missing or invalid."""


def canonical_key(value: Optional[str]) -> str:
    raw = str(value or "").strip().lower()
    return raw.replace(" ", "_").replace("-", "_")


def canonical_time_key(value: Optional[str]) -> str:
    key = canonical_key(value)
    return _TIME_ALIASES.get(key, key)


def canonical_weather_key(value: Optional[str]) -> str:
    return canonical_key(value)


def _require_str(obj: Dict[str, Any], key: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value.strip():
        raise TimeWeatherSpecError(f"Missing/invalid string field '{key}'")
    return value.strip()


def _require_float(obj: Dict[str, Any], key: str) -> float:
    try:
        return float(obj[key])
    except Exception as exc:
        raise TimeWeatherSpecError(f"Missing/invalid numeric field '{key}'") from exc


def _normalize_time_preset(item: Dict[str, Any]) -> Dict[str, Any]:
    key = canonical_time_key(_require_str(item, "key"))
    label = _require_str(item, "label")
    streetlights = canonical_key(_require_str(item, "streetlights"))
    if streetlights not in VALID_STREETLIGHT_MODES:
        raise TimeWeatherSpecError(
            f"Invalid streetlights mode '{streetlights}' for time preset '{key}'"
        )
    return {
        "key": key,
        "label": label,
        "sun_altitude_angle": _require_float(item, "sun_altitude_angle"),
        "sun_azimuth_angle": _require_float(item, "sun_azimuth_angle"),
        "streetlights": streetlights,
    }


def _normalize_weather_preset(item: Dict[str, Any]) -> Dict[str, Any]:
    key = canonical_weather_key(_require_str(item, "key"))
    label = _require_str(item, "label")
    return {
        "key": key,
        "label": label,
        "cloudiness": _require_float(item, "cloudiness"),
        "precipitation": _require_float(item, "precipitation"),
        "precipitation_deposits": _require_float(item, "precipitation_deposits"),
        "wind_intensity": _require_float(item, "wind_intensity"),
        "fog_density": _require_float(item, "fog_density"),
        "fog_distance": _require_float(item, "fog_distance"),
        "wetness": _require_float(item, "wetness"),
    }


def validate_and_normalize_spec(raw: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raise TimeWeatherSpecError("Spec root must be a JSON object")

    try:
        schema_version = int(raw.get("schema_version"))
    except Exception as exc:
        raise TimeWeatherSpecError("Missing/invalid 'schema_version'") from exc
    if schema_version != 1:
        raise TimeWeatherSpecError(
            f"Unsupported schema_version={schema_version}; expected 1"
        )

    matrix_output_subdir = str(raw.get("matrix_output_subdir", "")).strip()
    if not matrix_output_subdir:
        raise TimeWeatherSpecError("Missing/invalid 'matrix_output_subdir'")

    time_items = raw.get("time_presets")
    weather_items = raw.get("weather_presets")
    if not isinstance(time_items, list) or not time_items:
        raise TimeWeatherSpecError("Missing/invalid 'time_presets' list")
    if not isinstance(weather_items, list) or not weather_items:
        raise TimeWeatherSpecError("Missing/invalid 'weather_presets' list")

    time_presets: List[Dict[str, Any]] = []
    weather_presets: List[Dict[str, Any]] = []
    seen_time: set[str] = set()
    seen_weather: set[str] = set()

    for item in time_items:
        if not isinstance(item, dict):
            raise TimeWeatherSpecError("Each time preset must be an object")
        preset = _normalize_time_preset(item)
        if preset["key"] in seen_time:
            raise TimeWeatherSpecError(f"Duplicate time preset key '{preset['key']}'")
        seen_time.add(preset["key"])
        time_presets.append(preset)

    for item in weather_items:
        if not isinstance(item, dict):
            raise TimeWeatherSpecError("Each weather preset must be an object")
        preset = _normalize_weather_preset(item)
        if preset["key"] in seen_weather:
            raise TimeWeatherSpecError(f"Duplicate weather preset key '{preset['key']}'")
        seen_weather.add(preset["key"])
        weather_presets.append(preset)

    return {
        "schema_version": 1,
        "matrix_output_subdir": matrix_output_subdir,
        "time_presets": time_presets,
        "weather_presets": weather_presets,
    }


def load_time_weather_spec(spec_path: Optional[Union[Path, str]] = None) -> Dict[str, Any]:
    path = Path(spec_path) if spec_path else DEFAULT_SPEC_PATH
    if not path.exists():
        raise TimeWeatherSpecError(f"Spec file not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TimeWeatherSpecError(f"Invalid JSON spec: {path}: {exc}") from exc
    normalized = validate_and_normalize_spec(raw)
    normalized["spec_path"] = str(path.resolve())
    return normalized


def time_presets_by_key(spec: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(item["key"]): copy.deepcopy(item) for item in spec["time_presets"]}


def weather_presets_by_key(spec: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(item["key"]): copy.deepcopy(item) for item in spec["weather_presets"]}


def get_time_preset(spec: Dict[str, Any], key: Optional[str]) -> Optional[Dict[str, Any]]:
    canon = canonical_time_key(key)
    if not canon or canon == SCENE_DEFAULT_KEY:
        return None
    return time_presets_by_key(spec).get(canon)


def get_weather_preset(spec: Dict[str, Any], key: Optional[str]) -> Optional[Dict[str, Any]]:
    canon = canonical_weather_key(key)
    if not canon or canon == SCENE_DEFAULT_KEY:
        return None
    return weather_presets_by_key(spec).get(canon)


def variant_folder_name(time_key: str, weather_key: str) -> str:
    return f"{canonical_time_key(time_key)}__{canonical_weather_key(weather_key)}"


def iter_time_weather_combinations(spec: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    for time_preset in spec["time_presets"]:
        for weather_preset in spec["weather_presets"]:
            yield {
                "time_preset": copy.deepcopy(time_preset),
                "weather_preset": copy.deepcopy(weather_preset),
                "variant_name": variant_folder_name(
                    str(time_preset["key"]),
                    str(weather_preset["key"]),
                ),
            }


def build_scene_env_cli_args(
    time_preset: Dict[str, Any],
    weather_preset: Dict[str, Any],
) -> List[str]:
    """Build explicit CLI args so standalone scenes don't need hardcoded preset values."""
    return [
        "--time-preset",
        str(time_preset["key"]),
        "--weather-preset",
        str(weather_preset["key"]),
        "--sun-altitude",
        str(float(time_preset["sun_altitude_angle"])),
        "--sun-azimuth",
        str(float(time_preset["sun_azimuth_angle"])),
        "--streetlights",
        str(time_preset.get("streetlights", "auto")),
        "--cloudiness",
        str(float(weather_preset["cloudiness"])),
        "--precipitation",
        str(float(weather_preset["precipitation"])),
        "--precipitation-deposits",
        str(float(weather_preset["precipitation_deposits"])),
        "--wind-intensity",
        str(float(weather_preset["wind_intensity"])),
        "--fog-density",
        str(float(weather_preset["fog_density"])),
        "--fog-distance",
        str(float(weather_preset["fog_distance"])),
        "--wetness",
        str(float(weather_preset["wetness"])),
    ]
