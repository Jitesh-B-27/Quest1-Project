@echo off
REM Start the web UI at http://localhost:8000
uvicorn webapp.main:app --host 0.0.0.0 --port 8000
