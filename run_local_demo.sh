#!/bin/bash

# SCAI Detection System - Local Demo Script (Linux)
# This script demonstrates the complete system workflow

echo "=== SCAI Real-Time Detection System Demo ==="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

# Check if required files exist
if [ ! -f "api/app.py" ]; then
    echo "Error: api/app.py not found. Please run from project root directory."
    exit 1
fi

echo "1. Starting API Server..."
echo "   (This will run in the background)"

# Start API server in background
python3 api/app.py &
API_PID=$!

# Wait for API to start
echo "   Waiting for API to initialize..."
sleep 3

# Check if API is running
if ! curl -s http://localhost:8000/health > /dev/null; then
    echo "   Error: API server failed to start"
    kill $API_PID 2>/dev/null
    exit 1
fi

echo "   ✓ API Server started (PID: $API_PID)"
echo ""

echo "2. Starting Detection System..."

# Start detection system
RESPONSE=$(curl -s -X POST http://localhost:8000/start)
echo "   Response: $RESPONSE"

if echo "$RESPONSE" | grep -q "success"; then
    echo "   ✓ Detection system started successfully"
else
    echo "   ✗ Failed to start detection system"
    echo "   Stopping API server..."
    kill $API_PID 2>/dev/null
    exit 1
fi

# echo ""
# echo "3. Launching Omniverse Visualization..."
# echo "   (This will run in the background)"

# Start Omniverse visualization in background
# python3 viz/omniverse_app.py &
# VIZ_PID=$!

# echo "   ✓ Visualization started (PID: $VIZ_PID)"
# echo ""

# echo "4. Monitoring Detection Output..."
# echo "   Press Ctrl+C to stop the demo"
# echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Stopping demo..."
    
    # Stop detection system
    echo "   Stopping detection system..."
    curl -s -X POST http://localhost:8000/stop > /dev/null
    
    # Kill background processes
    echo "   Stopping visualization..."
    kill $VIZ_PID 2>/dev/null
    
    echo "   Stopping API server..."
    kill $API_PID 2>/dev/null
    
    echo "   ✓ Demo stopped"
    exit 0
}

# Set trap for cleanup
trap cleanup SIGINT SIGTERM

#wait for user input to stop
echo ""
echo "Demo is running..."
echo "Press Enter to stop the demo"
read -r

# Cleanup
cleanup

