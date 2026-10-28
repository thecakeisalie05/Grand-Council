@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
    echo Python launcher not found. Install Python 3.10 or newer from python.org.
    pause
    exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
    echo Creating isolated Python environment...
    py -3 -m venv .venv
    if errorlevel 1 goto :failed
)
if not exist ".venv\council-ready" (
    echo Installing Grand Council dependencies...
    ".venv\Scripts\python.exe" -m pip install -e .
    if errorlevel 1 goto :failed
    echo ready>".venv\council-ready"
)
echo.
".venv\Scripts\python.exe" "launch.py"
if errorlevel 1 goto :failed
exit /b 0
:failed
echo Grand Council could not start. Inspect the error above.
pause
exit /b 1
