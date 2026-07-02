@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

cd /d "%~dp0"

echo ==========================================
echo        DOC CHATBOT - STARTUP SCRIPT
echo ==========================================
echo.

REM --- First-time setup detection ---
set "FIRST_RUN=false"

if not exist "backend\.venv" (
    echo [SETUP] Backend virtual environment not found.
    set "FIRST_RUN=true"
)

if not exist "frontend\node_modules" (
    echo [SETUP] Frontend dependencies not found.
    set "FIRST_RUN=true"
)

if "%FIRST_RUN%"=="true" (
    echo.
    echo ==========================================
    echo    FIRST RUN - INSTALLING DEPENDENCIES
echo    This may take a few minutes...
    echo ==========================================
    echo.

    REM --- Backend Setup ---
    if not exist "backend\.venv" (
        echo [1/4] Creating Python virtual environment...
        python -m venv backend\.venv
        if errorlevel 1 (
            echo ERROR: Failed to create virtual environment. Make sure Python is installed.
            pause
            exit /b 1
        )
        echo [1/4] Virtual environment created.
    )

    echo [2/4] Installing backend dependencies...
    backend\.venv\Scripts\pip install --upgrade pip >nul 2>&1
    backend\.venv\Scripts\pip install -r backend\requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install backend dependencies.
        pause
        exit /b 1
    )
    echo [2/4] Backend dependencies installed.

    REM --- Frontend Setup ---
    echo [3/4] Installing frontend dependencies...
    cd frontend
    call npm install
    if errorlevel 1 (
        echo ERROR: Failed to install frontend dependencies. Make sure Node.js is installed.
        pause
        exit /b 1
    )
    cd ..
    echo [3/4] Frontend dependencies installed.

    echo [4/4] Setup complete!
    echo.
) else (
    echo [SETUP] Dependencies already installed. Skipping setup.
)

echo ==========================================
echo         STARTING BACKEND & FRONTEND
echo ==========================================
echo.

REM --- Start Backend ---
echo [START] Launching backend on http://localhost:8000 ...
start "Backend Server" cmd /c "cd backend && .venv\Scripts\activate && uvicorn app.api:app --reload --port 8000 && pause"

REM --- Start Frontend ---
echo [START] Launching frontend on http://localhost:5173 ...
start "Frontend Server" cmd /c "cd frontend && npm run dev && pause"

:: Wait for frontend to be ready, then open browser
timeout /t 5 /nobreak >nul
echo Opening frontend in browser...
start http://localhost:5173

echo.
echo ==========================================
echo      BOTH SERVERS ARE STARTING...
echo   Backend:   http://localhost:8000
echo   Frontend:  http://localhost:5173
echo ==========================================

pause
