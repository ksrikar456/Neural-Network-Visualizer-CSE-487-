@echo off
echo Starting Neural Style Transfer Application

REM Start the backend in a new window
start cmd /k "cd backend && python run.py"

REM Wait for backend to initialize
echo Waiting for backend to start...
timeout /t 5

REM Start the frontend
echo Starting frontend...
npm start

echo Development environment started. 