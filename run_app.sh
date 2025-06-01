#!/bin/bash

# Add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Check if port is already in use
if lsof -i:7860; then
    echo "Port 7860 is already in use. Attempting to free it..."
    kill $(lsof -t -i:7860) 2>/dev/null || true
    sleep 2
fi

# Start the application
echo "Starting Control Vector Application..."
python app.py &
APP_PID=$!

# Function to clean up on exit
cleanup() {
    echo "Shutting down application..."
    kill $APP_PID 2>/dev/null || true
    # Double check port is freed
    kill $(lsof -t -i:7860) 2>/dev/null || true
    echo "Cleanup complete."
    exit 0
}

# Set up trap for cleanup
trap cleanup SIGINT SIGTERM

# Wait for app process
wait $APP_PID
