

from flask import Flask, request, jsonify
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Tracker.position_estimator import PositionEstimator
from Tracker import config

app = Flask(__name__)
position_estimator: PositionEstimator = None
system_state = "stopped"  # Track system state: "stopped", "starting", "running", "stopping"

@app.route('/start', methods=['POST'])
def start():
    global position_estimator, system_state
    
    try:
        # Check current state
        if system_state == "running":
            return jsonify({
                'status': 'already_running',
                'message': 'Detection system is already running',
                'state': system_state
            }), 200
        
        if system_state == "starting":
            return jsonify({
                'status': 'starting',
                'message': 'Detection system is currently starting up',
                'state': system_state
            }), 200
        
        if system_state == "stopping":
            return jsonify({
                'status': 'error',
                'message': 'Detection system is currently stopping. Please wait.',
                'state': system_state
            }), 400
        
        # Start the system
        system_state = "starting"
        
        # Create new position estimator instance
        position_estimator = PositionEstimator()
        
        # Start all processes
        position_estimator.start()
        
        # Update state
        system_state = "running"

        return jsonify({
            'status': 'success',
            'message': 'Detection system started successfully',
            'state': system_state
        }), 200
    
    except Exception as e:
        system_state = "stopped"
        return jsonify({
            'status': 'error',
            'message': f'Error starting detection system: {str(e)}',
            'state': system_state
        }), 500

@app.route('/stop', methods=['POST'])
def stop():
    global position_estimator, system_state

    try:
        # Check current state
        if system_state == "stopped":
            return jsonify({
                'status': 'already_stopped',
                'message': 'Detection system is already stopped',
                'state': system_state
            }), 200
        
        if system_state == "stopping":
            return jsonify({
                'status': 'stopping',
                'message': 'Detection system is currently stopping',
                'state': system_state
            }), 200
        
        if system_state == "starting":
            return jsonify({
                'status': 'error',
                'message': 'Detection system is currently starting. Please wait.',
                'state': system_state
            }), 400
        
        # Stop the system
        system_state = "stopping"
        
        if position_estimator:
            position_estimator.stop()
            position_estimator = None
        
        system_state = "stopped"
        
        return jsonify({
            'status': 'success',
            'message': 'Detection system stopped successfully',
            'state': system_state
        }), 200
        
    except Exception as e:
        system_state = "stopped"
        return jsonify({
            'status': 'error',
            'message': f'Error stopping detection system: {str(e)}',
            'state': system_state
        }), 500

@app.route('/read', methods=['GET'])
def read():
    global position_estimator
    
    if position_estimator is None:
        return jsonify([])
    
    # Get detections - simple approach
    detections = position_estimator.get_detections()
    return jsonify(detections)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"message": "OK"}), 200

@app.route('/status', methods=['GET'])
def status():
    global position_estimator, system_state
    
    # Get detection count if system is running
    detection_count = 0
    if position_estimator and system_state == "running":
        try:
            detections = position_estimator.get_detections()
            detection_count = len(detections) if detections else 0
        except:
            detection_count = 0
    
    return jsonify({
        'state': system_state,
        'detection_count': detection_count,
        'system_running': system_state == "running"
    }), 200

@app.route('/stats', methods=['GET'])
def stats():
    global position_estimator
    
    if position_estimator is None:
        return jsonify({"error": "System not started"}), 400
    
    try:
        stats = position_estimator.get_performance_stats()
        return jsonify(stats), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Run the Flask app
    app.run(
        host='127.0.0.1',  # Allow external access
        port=5000,
        debug=config.verbose,
        threaded=True
    )