"""Snow-oriented weather helpers for generated CARLA scenes.

These helpers support two modes:
1) Native snow builds (patched CARLA exposing WeatherParameters.snow)
2) Fallback mode for stock builds using precipitation/fog/wetness/wind
"""

from __future__ import annotations

from typing import Any


def supports_native_snow(carla_module: Any) -> bool:
    """Return True when the runtime WeatherParameters struct exposes `snow`."""
    try:
        weather = carla_module.WeatherParameters()
        return hasattr(weather, "snow")
    except Exception:
        return False


def build_snow_fallback_weather(carla_module: Any, intensity: float = 80.0) -> Any:
    """Build a stock-CARLA winter storm profile without native `snow` support."""
    i = max(0.0, min(100.0, float(intensity)))
    precip = min(100.0, 20.0 + 0.6 * i)
    deposits = min(100.0, 35.0 + 0.65 * i)
    fog = min(100.0, 15.0 + 0.5 * i)
    wind = min(100.0, 10.0 + 0.7 * i)
    wet = min(100.0, 30.0 + 0.7 * i)

    return carla_module.WeatherParameters(
        cloudiness=min(100.0, 35.0 + 0.6 * i),
        precipitation=precip,
        precipitation_deposits=deposits,
        wind_intensity=wind,
        sun_azimuth_angle=95.0,
        sun_altitude_angle=8.0,
        fog_density=fog,
        fog_distance=max(10.0, 120.0 - i),
        fog_falloff=0.2,
        wetness=wet,
        scattering_intensity=1.0,
        mie_scattering_scale=0.03,
        rayleigh_scattering_scale=0.0331,
    )


def build_native_snow_weather(carla_module: Any, intensity: float = 80.0) -> Any:
    """Build weather for patched CARLA exposing WeatherParameters.snow."""
    i = max(0.0, min(100.0, float(intensity)))
    weather = build_snow_fallback_weather(carla_module, intensity=i)
    try:
        weather.snow = i
    except Exception:
        # Keep fallback weather when runtime does not accept .snow assignments.
        pass
    return weather


def apply_snow_weather(world: Any, carla_module: Any, intensity: float = 80.0) -> str:
    """Apply snow-like weather and return mode string: native_snow or fallback."""
    if supports_native_snow(carla_module):
        world.set_weather(build_native_snow_weather(carla_module, intensity))
        return "native_snow"
    world.set_weather(build_snow_fallback_weather(carla_module, intensity))
    return "fallback"

