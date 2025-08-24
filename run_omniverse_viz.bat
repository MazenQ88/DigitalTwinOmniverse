@echo off
setlocal

REM Omniverse Detection Visualization Launcher with WebRTC Remote Rendering
REM This script runs the detection visualizer in Omniverse Kit with remote streaming support

echo === Omniverse Detection Visualizer with WebRTC Remote Rendering ===
echo.

REM Set the path to kit.exe (adjust this to your Omniverse installation path)
set KIT_PATH=kit.exe

REM WebRTC Configuration
set WEBRTC_PORT=8211
set WEBRTC_INTERFACE=0.0.0.0
set ENABLE_WEBRTC=true

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

echo Starting Omniverse Kit with Detection Visualizer and WebRTC Remote Rendering...
echo Script: %SCRIPT_PATH%
if not "%USD_FILE%"=="" (
    echo USD File: %USD_FILE%
)
if "%ENABLE_WEBRTC%"=="true" (
    echo WebRTC Remote Rendering: ENABLED
    echo WebRTC Port: %WEBRTC_PORT%
    echo WebRTC Interface: %WEBRTC_INTERFACE%
    echo Remote Access URL: http://localhost:%WEBRTC_PORT%
)
echo.

REM Build kit command with WebRTC extensions if enabled
set KIT_ARGS=--enable omni.kit.uiapp --enable omni.usd --enable omni.ui --enable omni.kit.commands --enable omni.kit.widget.viewport --enable omni.kit.viewport.window --enable omni.kit.viewport.actions --enable omni.kit.viewport.utility --enable omni.kit.viewport.rtx --enable omni.hydra.rtx --enable omni.kit.manipulator.camera --enable omni.kit.manipulator.prim --enable omni.kit.manipulator.selection --enable omni.kit.manipulator.viewport --enable omni.kit.manipulator.transform --enable omni.kit.viewport.legacy_gizmos --enable omni.kit.window.stage --enable omni.kit.window.property --enable omni.kit.window.content_browser --enable omni.kit.property.usd --enable omni.kit.selection --enable omni.kit.stage_templates --enable omni.kit.widget.stage --enable omni.kit.widget.layers --enable omni.kit.context_menu --enable omni.kit.hotkeys.window --enable omni.kit.primitive.mesh --enable omni.kit.window.console

REM Add WebRTC streaming extensions if enabled
if "%ENABLE_WEBRTC%"=="true" (
    echo Enabling WebRTC remote streaming...
    echo WebRTC will be available at: http://localhost:%WEBRTC_PORT%/streaming/webrtc-client?server=localhost
    REM Current WebRTC extensions for Kit 106.4+ (replaces deprecated omni.services.streamclient.webrtc)
    set KIT_ARGS=%KIT_ARGS% --enable omni.kit.livestream.webrtc --enable omni.kit.livestream.webrtc.setup --enable omni.services.transport.server.http
    REM WebRTC streaming configuration
    set KIT_ARGS=%KIT_ARGS% --/app/livestream/enabled=true --/app/livestream/port=%WEBRTC_PORT% --/app/livestream/webrtc/enabled=true --/app/livestream/webrtc/port=%WEBRTC_PORT% --/app/livestream/webrtc/interface=%WEBRTC_INTERFACE% --/app/window/hideUi=false
) else (
    echo WebRTC remote streaming is disabled.
    echo To enable: set ENABLE_WEBRTC=true in this script.
)

REM Add script execution
set KIT_ARGS=%KIT_ARGS% --exec "%SCRIPT_PATH% %USD_FILE%"

REM Run kit.exe with all arguments
"%KIT_PATH%" %KIT_ARGS%

if errorlevel 1 (
    echo.
    echo Kit exited with error code %errorlevel%
    pause
)

exit /b 0
