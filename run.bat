@echo off
title CarnetLM
cd /d "%~dp0"

echo.
echo   ____                      _   _     __  __ 
echo  / ___|__ _ _ __ _ __   ___| |_| |   |  \/  |
echo | |   / _` | '__| '_ \ / _ \ __| |   | |\/| |
echo | |__| (_| | |  | | | |  __/ |_| |___| |  | |
echo  \____\__,_|_|  |_| |_|\___|\__|\_____|_|  |_|
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

:: Sync/Install all dependencies declared in pyproject.toml
echo Syncing and installing project dependencies...
uv sync
if %ERRORLEVEL% neq 0 (
    echo [WARNING] 'uv sync' failed. Trying fallback 'uv pip install -r requirements.txt'...
    uv pip install -r requirements.txt
)
echo.

echo Starting CarnetLM on http://localhost:8000
echo Press Ctrl+C to stop.
echo.

:: Run background browser opener
start /b cmd /c "timeout /t 3 >nul && start "" http://localhost:8000"

:: Start the server in the foreground
uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000

echo.
echo [CarnetLM stopped]
echo.
pause
