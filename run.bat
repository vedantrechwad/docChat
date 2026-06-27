@echo off
title DocChat
cd /d "%~dp0"

echo.
echo  ____             ____ _           _
echo ^|  _ \  ___   ___/ ___^| ^|__   __ _^| ^|_
echo ^| ^| ^| ^|/ _ \ / __^| ^|   ^| '_ \ / _` ^| __^|
echo ^| ^|_^| ^| (_) ^| (__^| ^|___^| ^| ^| ^| (_^| ^| ^|_
echo ^|____/ \___/ \___\____^|_^| ^|_^|\__,_^\__^|
echo.

:: Check if uv is available
where uv >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] 'uv' is not installed or not in PATH.
    echo Install it from: https://docs.astral.sh/uv/
    echo.
    pause
    exit /b 1
)

:: Kill any existing server on port 8000
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000.*LISTENING" 2^>nul') do (
    taskkill /PID %%a /F >nul 2>&1
)

echo Starting DocChat on http://localhost:8000
echo Press Ctrl+C to stop.
echo.

:: Start the server in background and open browser
start "" http://localhost:8000
timeout /t 2 /nobreak >nul

:: Start the server (stays open so you can see output)
uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000

:: If server exits, show the error and wait
echo.
echo [DocChat stopped]
echo.
pause
