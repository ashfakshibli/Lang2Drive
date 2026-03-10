@echo off
setlocal

cd /d "%~dp0"

if "%~1"=="" (
  echo [INFO] Running 20-combination time^xweather matrix for latest manifest pointer...
  python "%~dp0agentic_wine_time_weather_matrix_runner.py"
) else (
  echo [INFO] Running 20-combination time^xweather matrix for manifest:
  echo        %~1
  python "%~dp0agentic_wine_time_weather_matrix_runner.py" --handoff-manifest "%~1"
)

set EXIT_CODE=%ERRORLEVEL%

if %EXIT_CODE% NEQ 0 (
  echo.
  echo Matrix Stage 2 failed. Check handoff simulation_result_matrix20.json and stage2_runner_matrix20 log.
)

exit /b %EXIT_CODE%
