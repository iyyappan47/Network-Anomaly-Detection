@echo off
setlocal enabledelayedexpansion
title Network Anomaly Detection System

cd /d "%~dp0"

echo ======================================================================
echo           MACHINE LEARNING-BASED NETWORK ANOMALY DETECTION
echo ======================================================================
echo.

:: 1. Verify Python availability
set "PYTHON_CMD="
where python >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set "PYTHON_CMD=python"
) else (
    where py >nul 2>&1
    if %ERRORLEVEL% equ 0 (
        set "PYTHON_CMD=py -3"
    )
)

if "%PYTHON_CMD%"=="" (
    echo [ERROR] Python is not found in your system PATH.
    echo Please install Python 3.10+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

echo [1/3] Checking Python runtime...
%PYTHON_CMD% --version
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to execute Python command: %PYTHON_CMD%
    pause
    exit /b 1
)

:: 2. Verify dependencies
echo.
echo [2/3] Verifying required Python packages...
%PYTHON_CMD% -c "import fastapi, uvicorn, sklearn, pandas, numpy, joblib, requests" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [WARNING] Some required dependencies are missing.
    echo Installing dependencies from requirements.txt...
    %PYTHON_CMD% -m pip install -r requirements.txt
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to install required dependencies.
        pause
        exit /b 1
    )
)
echo [OK] All required packages are available.

:: 3. Launch application server
echo.
echo [3/3] Starting Network Anomaly Detection Server...
echo.
echo ======================================================================
echo  Dashboard URL:   http://127.0.0.1:8000
echo  API Docs:        http://127.0.0.1:8000/docs
echo  Health Status:   http://127.0.0.1:8000/api/health
echo ======================================================================
echo.
echo Press Ctrl+C in this console to stop the server at any time.
echo.

:: Attempt to open the default web browser after a brief delay
start "" cmd /c "timeout /t 2 /nobreak >nul & start http://127.0.0.1:8000"

:: Launch uvicorn
%PYTHON_CMD% -m uvicorn app:app --host 127.0.0.1 --port 8000

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Server terminated unexpectedly with exit code %ERRORLEVEL%.
    pause
    exit /b %ERRORLEVEL%
)

pause
