#!/bin/bash
echo "Starting Neural Style Transfer Application"

# Start the backend in a new terminal
echo "Starting backend..."
cd backend && python run.py &

# Wait for backend to initialize
echo "Waiting for backend to start..."
sleep 5

# Start the frontend
echo "Starting frontend..."
npm start

echo "Development environment started." 