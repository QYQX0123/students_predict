@echo off
set PYTHON_EXE=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe
if exist "%PYTHON_EXE%" (
    "%PYTHON_EXE%" run_app.py
) else (
    python run_app.py
)
