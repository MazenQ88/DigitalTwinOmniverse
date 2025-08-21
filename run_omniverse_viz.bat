@echo off
setlocal

REM Omniverse Detection Visualization Launcher
REM This script runs the detection visualizer in Omniverse Kit

echo === Omniverse Detection Visualizer ===
echo.

REM Set the path to kit.exe (adjust this to your Omniverse installation path)
set KIT_PATH=kit.exe

@REM REM Check if kit.exe exists
@REM if not exist "%KIT_PATH%" (
@REM     echo Error: kit.exe not found at %KIT_PATH%
@REM     echo Please update the KIT_PATH variable in this script to point to your kit.exe
@REM     echo.
@REM     echo Common locations:
@REM     echo - C:\Users\%USERNAME%\AppData\Local\ov\pkg\code-2023.2.2\kit\kit.exe
@REM     echo - C:\Users\%USERNAME%\AppData\Local\ov\pkg\create-2023.2.2\kit\kit.exe
@REM     echo - C:\Program Files\NVIDIA\Omniverse\kit\kit.exe
@REM     pause
@REM     exit /b 1
@REM )

REM Set paths
set SCRIPT_PATH=%~dp0Viz\omniverse_app.py
set USD_FILE=%~dp0Assets\city.usd

REM Check if the script exists
if not exist "%SCRIPT_PATH%" (
    echo Error: omniverse_app.py not found at %SCRIPT_PATH%
    pause
    exit /b 1
)

REM Check if USD file exists (optional - will create scene if not found)
if not exist "%USD_FILE%" (
    echo Warning: USD file not found at %USD_FILE%
    echo Will create a procedural scene instead.
    set USD_FILE=
)

echo Starting Omniverse Kit with Detection Visualizer...
echo Script: %SCRIPT_PATH%
if not "%USD_FILE%"=="" (
    echo USD File: %USD_FILE%
)
echo.

REM Run kit.exe with required extensions and execute the script
"%KIT_PATH%" ^
    --enable omni.kit.uiapp ^
    --enable omni.usd ^
    --enable omni.ui ^
    --enable omni.kit.commands ^
    --enable omni.kit.widget.viewport ^
    --enable omni.kit.viewport.window ^
    --enable omni.kit.viewport.actions ^
    --enable omni.kit.viewport.utility ^
    --enable omni.kit.viewport.rtx ^
    --enable omni.hydra.rtx ^
    --enable omni.hydra.rtx.shadercache.d3d12 ^
    --enable omni.kit.manipulator.camera ^
    --enable omni.kit.manipulator.prim ^
    --enable omni.kit.manipulator.selection ^
    --enable omni.kit.manipulator.viewport ^
    --enable omni.kit.manipulator.transform ^
    --enable omni.kit.viewport.legacy_gizmos ^
    --enable omni.kit.window.stage ^
    --enable omni.kit.window.property ^
    --enable omni.kit.window.content_browser ^
    --enable omni.kit.property.usd ^
    --enable omni.kit.selection ^
    --enable omni.kit.stage_templates ^
    --enable omni.kit.widget.stage ^
    --enable omni.kit.widget.layers ^
    --enable omni.kit.context_menu ^
    --enable omni.kit.hotkeys.window ^
    --enable omni.kit.primitive.mesh ^
    --enable omni.kit.window.console ^
    --exec "%SCRIPT_PATH% %USD_FILE%"

if errorlevel 1 (
    echo.
    echo Kit exited with error code %errorlevel%
    pause
)

exit /b 0
