@echo off
echo.
echo  ____             ____ _           _
echo ^|  _ \  ___   ___/ ___^| ^|__   __ _^| ^|_
echo ^| ^| ^| ^|/ _ \ / __^| ^|   ^| '_ \ / _` ^| __^|
echo ^| ^|_^| ^| (_) ^| (__^| ^|___^| ^| ^| ^| (_^| ^| ^|_
echo ^|____/ \___/ \___\____^|_^| ^|_^|\__,_^|\__^|
echo.
echo Starting DocChat on http://localhost:8000
echo Press Ctrl+C to stop.
echo.
uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000
