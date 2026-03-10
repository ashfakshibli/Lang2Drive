@echo off
echo =========================================
echo CARLA Car Camera Simulation Launcher
echo =========================================
echo.

REM Use Python from PATH
set PYTHON_CMD=python

REM Default values
set DURATION=60
set OUTPUT_DIR=output_images

REM Check if custom duration provided
if not "%1"=="" set DURATION=%1
if not "%2"=="" set OUTPUT_DIR=%2

echo Starting simulation with:
echo - Duration: %DURATION% seconds
echo - Output directory: %OUTPUT_DIR%
echo.
echo Press Ctrl+C to stop early
echo.

REM Run the simulation
%PYTHON_CMD% car_camera_simulation.py --duration %DURATION% --output-dir "%OUTPUT_DIR%"

echo.
echo Simulation completed!
pause
