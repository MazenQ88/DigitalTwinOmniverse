#!/bin/bash

# Omniverse Detection Visualization Launcher with WebRTC Remote Rendering
# This script runs the detection visualizer in Omniverse Kit with remote streaming support

echo "=== Omniverse Detection Visualizer with WebRTC Remote Rendering ==="
echo

# Set the path to kit executable (adjust this to your Omniverse installation path)
KIT_PATH="kit"

# WebRTC Configuration
WEBRTC_PORT=8211
WEBRTC_INTERFACE="0.0.0.0"
ENABLE_WEBRTC=true

# Common locations for Omniverse Kit on different platforms
# Uncomment and modify the appropriate path for your system:

# macOS (typical locations)
# KIT_PATH="/Applications/Omniverse/Kit/kit"
# KIT_PATH="$HOME/Library/Application Support/ov/pkg/code-2023.2.2/kit/kit"

# Linux (typical locations)
# KIT_PATH="$HOME/.local/share/ov/pkg/code-2023.2.2/kit/kit"
# KIT_PATH="/opt/nvidia/omniverse/kit/kit"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set paths
SCRIPT_PATH="$SCRIPT_DIR/Viz/omniverse_app.py"
USD_FILE="$SCRIPT_DIR/Assets/city.usd"

# Check if the script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: omniverse_app.py not found at $SCRIPT_PATH"
    exit 1
fi

# Check if USD file exists (optional - will create scene if not found)
if [ ! -f "$USD_FILE" ]; then
    echo "Warning: USD file not found at $USD_FILE"
    echo "Will create a procedural scene instead."
    USD_FILE=""
fi

echo "Starting Omniverse Kit with Detection Visualizer and WebRTC Remote Rendering..."
echo "Script: $SCRIPT_PATH"
if [ -n "$USD_FILE" ]; then
    echo "USD File: $USD_FILE"
fi
if [ "$ENABLE_WEBRTC" = true ]; then
    echo "WebRTC Remote Rendering: ENABLED"
    echo "WebRTC Port: $WEBRTC_PORT"
    echo "WebRTC Interface: $WEBRTC_INTERFACE"
    echo "Remote Access URL: http://localhost:$WEBRTC_PORT"
fi
echo

# Check if kit executable exists
if ! command -v "$KIT_PATH" >/dev/null 2>&1; then
    echo "Error: kit executable not found at $KIT_PATH"
    echo "Please update the KIT_PATH variable in this script to point to your kit executable"
    echo
    echo "Common locations:"
    echo "macOS:"
    echo "  - /Applications/Omniverse/Kit/kit"
    echo "  - \$HOME/Library/Application Support/ov/pkg/code-2023.2.2/kit/kit"
    echo "Linux:"
    echo "  - \$HOME/.local/share/ov/pkg/code-2023.2.2/kit/kit"
    echo "  - /opt/nvidia/omniverse/kit/kit"
    exit 1
fi

# Build kit command with WebRTC extensions if enabled
KIT_ARGS=(
    --enable omni.kit.uiapp
    --enable omni.usd
    --enable omni.ui
    --enable omni.kit.commands
    --enable omni.kit.widget.viewport
    --enable omni.kit.viewport.window
    --enable omni.kit.viewport.actions
    --enable omni.kit.viewport.utility
    --enable omni.kit.viewport.rtx
    --enable omni.hydra.rtx
    --enable omni.kit.manipulator.camera
    --enable omni.kit.manipulator.prim
    --enable omni.kit.manipulator.selection
    --enable omni.kit.manipulator.viewport
    --enable omni.kit.manipulator.transform
    --enable omni.kit.viewport.legacy_gizmos
    --enable omni.kit.window.stage
    --enable omni.kit.window.property
    --enable omni.kit.window.content_browser
    --enable omni.kit.property.usd
    --enable omni.kit.selection
    --enable omni.kit.stage_templates
    --enable omni.kit.widget.stage
    --enable omni.kit.widget.layers
    --enable omni.kit.context_menu
    --enable omni.kit.hotkeys.window
    --enable omni.kit.primitive.mesh
    --enable omni.kit.window.console
)

# Add WebRTC streaming extensions if enabled
if [ "$ENABLE_WEBRTC" = true ]; then
    echo "Enabling WebRTC remote streaming..."
    echo "WebRTC will be available at: http://localhost:$WEBRTC_PORT/streaming/webrtc-client?server=localhost"
    
    KIT_ARGS+=(
        # Current WebRTC extensions for Kit 106.4+ (replaces deprecated omni.services.streamclient.webrtc)
        --enable omni.kit.livestream.webrtc
        --enable omni.kit.livestream.webrtc.setup
        --enable omni.services.transport.server.http
        
        # WebRTC streaming configuration
        --/app/livestream/enabled=true
        --/app/livestream/port=$WEBRTC_PORT
        --/app/livestream/webrtc/enabled=true
        --/app/livestream/webrtc/port=$WEBRTC_PORT
        --/app/livestream/webrtc/interface="$WEBRTC_INTERFACE"
        --/app/window/hideUi=false
    )
else
    echo "WebRTC remote streaming is disabled."
    echo "To enable: set ENABLE_WEBRTC=true in this script."
fi

# Add script execution
KIT_ARGS+=(--exec "$SCRIPT_PATH $USD_FILE")

# Run kit with all arguments
"$KIT_PATH" "${KIT_ARGS[@]}"

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo
    echo "Kit exited with error code $EXIT_CODE"
    read -p "Press any key to continue..."
fi

exit $EXIT_CODE
