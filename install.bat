@echo off
REM ============================================================
REM MouseReach Installation Script
REM ============================================================
REM Run this from Anaconda Prompt after copying MouseReach to your machine.
REM
REM Usage:
REM   1. Open Anaconda Prompt (or PowerShell with conda)
REM   2. cd to the MouseReach folder
REM   3. Run: install.bat
REM
REM Creates:
REM   - Conda environment named 'mousereach'
REM   - Data pipeline folders at configured location
REM ============================================================

echo.
echo ============================================================
echo MouseReach INSTALLER v2.3
echo ============================================================
echo.

REM Check if conda is available
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: conda not found!
    echo.
    echo Please run this from Anaconda Prompt, not regular Command Prompt.
    echo.
    echo To open Anaconda Prompt:
    echo   1. Press Windows key
    echo   2. Type "Anaconda Prompt"
    echo   3. Click to open
    echo   4. cd to this folder and run install.bat again
    echo.
    pause
    exit /b 1
)

REM Get the directory where this script is located
set SCRIPT_DIR=%~dp0
set SCRIPT_DIR=%SCRIPT_DIR:~0,-1%

echo.
echo Step 1/4: Creating conda environment 'mousereach'...
echo.
call conda create -n mousereach python=3.10 -y
if %errorlevel% neq 0 (
    echo Environment may already exist. Trying to activate...
)

echo.
echo Step 2/4: Activating environment...
echo.
call conda activate mousereach
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate environment
    pause
    exit /b 1
)

echo.
echo Step 3/4: Installing MouseReach and all dependencies...
echo (This may take 10-15 minutes on first install)
echo.
pip install -e "%SCRIPT_DIR%"
if %errorlevel% neq 0 (
    echo ERROR: Failed to install MouseReach
    pause
    exit /b 1
)

REM Pin numpy for TensorFlow compatibility
pip install numpy==1.26.4 -q

echo.
echo Step 4/4: Verifying installation...
echo.
python -c "import deeplabcut; print('DeepLabCut: OK')" 2>nul || echo DeepLabCut: NOT INSTALLED (install separately via conda)
python -c "import napari; print('Napari: OK')"
python -c "import cv2; print('OpenCV: OK')"
python -c "from mousereach.launcher import main; print('MouseReach: OK')"

echo.
echo ============================================================
echo INSTALLATION COMPLETE!
echo ============================================================
echo.
echo To use MouseReach:
echo   1. Open Anaconda Prompt
echo   2. Run: conda activate mousereach
echo   3. Run: mousereach
echo.
echo Or with a video:
echo   mousereach "path\to\video.mp4"
echo.
echo Run 'mousereach-setup' to configure pipeline paths.
echo See README.md for detailed usage instructions.
echo ============================================================
echo.
pause
