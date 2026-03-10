@echo off
setlocal

cd /d "%~dp0"
python "%~dp0agentic_wine_handoff_runner.py"
set EXIT_CODE=%ERRORLEVEL%

if %EXIT_CODE% NEQ 0 (
  echo.
  echo Stage 2 failed. Check handoff simulation_result.json for details.
)

exit /b %EXIT_CODE%
