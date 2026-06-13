@echo off
echo ===================================================
echo     Starting Study Companion
echo ===================================================

:: Start Ollama Server in the background
echo Starting Ollama Server...
start "Ollama Server" /MIN cmd /k "ollama serve"

:: Wait briefly to ensure Ollama initializes
timeout /t 2 /nobreak > nul

:: Start the FastAPI backend in a new window
echo Starting FastAPI Backend...
start "Study Companion Backend" cmd /k "uv run uvicorn backend.main:app --reload"

:: Wait for a couple of seconds to ensure backend starts
timeout /t 3 /nobreak > nul

:: Start the Next.js frontend in a new window
echo Starting Next.js Frontend...
start "Study Companion Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo ===================================================
echo Both services are starting up!
echo Backend: http://localhost:8000
echo Frontend: http://localhost:3000
echo ===================================================
pause
