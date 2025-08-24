#!/bin/bash

# SCAI Detection System - Local Demo Script (Unix/Linux/macOS)
# This script demonstrates the complete system workflow

set -e  # Exit on any error

echo "=== Real-Time Detection System Demo ==="
echo

# Function to cleanup on exit
cleanup() {
    echo
    echo "Stopping demo..."
    
    # Stop detection system
    echo "    Stopping detection system..."
    curl -s -X POST http://localhost:5000/stop >/dev/null 2>&1 || true
    
    # Kill Python processes started by this script
    echo "    Stopping API server and visualization..."
    if [ -n "$API_PID" ]; then
        kill $API_PID >/dev/null 2>&1 || true
    fi
    
    # Clean up temporary files
    rm -f start_response.json >/dev/null 2>&1 || true
    
    echo "    ✓ Demo stopped"
}

# Set trap to cleanup on script exit
trap cleanup EXIT

# Check if Python is available (try python3 first, then python)
PYTHON=""
if command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON="python"
else
    echo "Error: Python 3 is required but not installed."
    echo "Tried: python3, python"
    exit 1
fi

# Check if required files exist
if [ ! -f "API/app.py" ]; then
    echo "Error: API/app.py not found. Please run from project root directory."
    exit 1
fi

# Check if curl is available
if ! command -v curl >/dev/null 2>&1; then
    echo "Error: curl is required but not installed."
    echo "Please install curl using your package manager."
    exit 1
fi

echo "1. Starting API Server..."
echo "   (This will run in the background)"
echo "   Using Python command: $PYTHON"

# Start API server in background
$PYTHON API/app.py >/dev/null 2>&1 &
API_PID=$!

# Wait for API to start
echo "   Waiting for API to initialize..."
sleep 5

# Check if API is running
echo "   Testing API connection..."
if ! curl -s http://localhost:5000/health >/dev/null 2>&1; then
    echo "   Error: API server failed to start"
    exit 1
fi

echo "   ✓ API Server started"
echo

echo "2. Starting Detection System..."

# Start detection system with better error handling
echo "   Sending start request to API..."
if ! curl -s -X POST http://localhost:5000/start -o start_response.json 2>&1; then
    echo "   × Failed to communicate with API server"
    echo "   Stopping API server..."
    exit 1
fi

# Read the response
if [ -f "start_response.json" ]; then
    echo "   --- API Response ---"
    cat start_response.json
    echo
    echo "   --- End Response ---"
    
    # Check if response contains success
    if grep -q "success" start_response.json; then
        echo "   ✓ Detection system started successfully"
        rm -f start_response.json
    else
        echo "   × Failed to start detection system"
        echo "   Check the API response above for details"
        rm -f start_response.json
        echo "   Stopping API server..."
        exit 1
    fi
else
    echo "   × No response received from API"
    echo "   Stopping API server..."
    exit 1
fi

echo
echo "3. Launching Omniverse Visualization..."
echo "   (This will open Omniverse Kit)"
echo

# Check if run_omniverse_viz.sh exists
if [ ! -f "run_omniverse_viz.sh" ]; then
    echo "   Error: run_omniverse_viz.sh not found"
    echo "   Stopping API server..."
    exit 1
fi

# Make sure the script is executable
chmod +x run_omniverse_viz.sh

# Start Omniverse visualization
echo "   Starting Omniverse Kit..."
./run_omniverse_viz.sh

echo
echo "Omniverse visualization has closed."
echo "Press any key to stop the demo and cleanup..."
read -n 1 -s

# Cleanup will be handled by the trap
exit 0
