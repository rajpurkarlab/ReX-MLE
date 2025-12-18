#!/bin/bash
# Startup script for ML-Master grading server

# Check if server is already running
if pgrep -f "grading_server.py" > /dev/null; then
    echo "Grading server is already running."
    PID=$(pgrep -f "grading_server.py")
    echo "PID: $PID"
    curl -s http://127.0.0.1:5001/health
    exit 0
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Start the grading server
echo "Starting grading server on port 5001..."
python grading_server.py > logs/grading_server.log 2>&1 &
PID=$!

# Wait for server to start
sleep 3

# Check if server is running
if pgrep -f "grading_server.py" > /dev/null; then
    echo "Grading server started successfully!"
    echo "PID: $PID"
    echo "Log file: logs/grading_server.log"

    # Test health endpoint
    HEALTH=$(curl -s http://127.0.0.1:5001/health)
    echo "Health check: $HEALTH"
else
    echo "Failed to start grading server. Check logs/grading_server.log for errors."
    exit 1
fi
