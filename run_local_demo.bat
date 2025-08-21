@echo off
setlocal enabledelayedexpansion

REM SCAI Detection System - Local Demo Script (Windows)
REM This script demonstrates the complete system workflow

echo ===  Real-Time Detection System Demo ===
echo.

REM Check if Python is available (try python3 first, then python, then py)
where python3 >nul 2>nul
if errorlevel 1 (
    where python >nul 2>nul
    if errorlevel 1 (
        where py >nul 2>nul
        if errorlevel 1 (
            echo Error: Python 3 is required but not installed.
            echo Tried: python3, python, py
            pause
            exit /b 1
        ) else (
            set PYTHON=py
        )
    ) else (
        set PYTHON=python
    )
) else (
    set PYTHON=python3
)

REM Check if required files exist
if not exist "API\app.py" (
    echo Error: API\app.py not found. Please run from project root directory.
    pause
    exit /b 1
)

REM Check if curl is available
where curl >nul 2>nul
if errorlevel 1 (
    echo Error: curl is required but not installed.
    echo Please install curl or use Windows 10 version 1803 or later.
    pause
    exit /b 1
)

echo 1. Starting API Server...
echo    ^(This will run in the background^)
echo    Using Python command: %PYTHON%

REM Start API server in background
start /B cmd /c "%PYTHON% API\app.py 2>nul"

REM Wait for API to start
echo    Waiting for API to initialize...
timeout /t 5 /nobreak >nul

REM Check if API is running
echo    Testing API connection...
curl -s http://localhost:8000/health >nul 2>&1
if errorlevel 1 (
    echo    Error: API server failed to start
    call :cleanup
    exit /b 1
)

echo    √ API Server started
echo.

echo 2. Starting Detection System...

REM Start detection system with better error handling
echo    Sending start request to API...
curl -s -X POST http://localhost:8000/start -o start_response.json 2>&1
if errorlevel 1 (
    echo    × Failed to communicate with API server
    echo    Stopping API server...
    call :cleanup
    exit /b 1
)

REM Read the response
if exist "start_response.json" (
    echo    --- API Response ---
    type start_response.json
    echo.
    echo    --- End Response ---
    
    REM Check if response contains success
    findstr /C:"success" start_response.json >nul
    if %errorlevel%==0 (
        echo    √ Detection system started successfully
        del start_response.json >nul 2>&1
    ) else (
        echo    × Failed to start detection system
        echo    Check the API response above for details
        del start_response.json >nul 2>&1
        echo    Stopping API server...
        call :cleanup
        exit /b 1
    )
) else (
    echo    × No response received from API
    echo    Stopping API server...
    call :cleanup
    exit /b 1
)

echo.
echo 3. Launching Omniverse Visualization...
echo    ^(This will open Omniverse Kit^)
echo.

REM Check if run_omniverse_viz.bat exists
if not exist "run_omniverse_viz.bat" (
    echo    Error: run_omniverse_viz.bat not found
    echo    Stopping API server...
    call :cleanup
    exit /b 1
)

REM Start Omniverse visualization
echo    Starting Omniverse Kit...
call run_omniverse_viz.bat

echo.
echo Omniverse visualization has closed.
echo Press any key to stop the demo and cleanup...
pause >nul

REM Cleanup
call :cleanup
exit /b 0

:cleanup
echo.
echo Stopping demo...

REM Stop detection system
echo    Stopping detection system...
curl -s -X POST http://localhost:8000/stop >nul 2>&1

REM Kill Python processes (be careful with this - it will kill ALL Python processes)
echo    Stopping API server and visualization...
REM This will kill all Python processes started by this script
for /f "tokens=2" %%i in ('tasklist ^| findstr /i python') do (
    taskkill /PID %%i /F >nul 2>&1
)

REM Clean up temporary files
if exist "start_response.json" del start_response.json >nul 2>&1

echo    √ Demo stopped
exit /b 0
