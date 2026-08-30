@echo off
REM ============================================================
REM MouseReach Watcher - Auto-launch on login
REM ============================================================
REM Starts the watcher daemon and a status monitor in separate windows.
REM Close the watcher window (or Ctrl+C) to stop processing.
REM
REM Install: copy this file to your Startup folder:
REM   Win+R -> shell:startup -> paste this file
REM ============================================================

title MouseReach Launcher

REM --- Auto-detect the mousereach-watch executable ---
REM Check common environment locations in order of preference
set "WATCH_EXE="
for %%P in (
    "%MOUSEREACH_ENV%\Scripts\mousereach-watch.exe"
    "%USERPROFILE%\miniconda3\envs\mousereach\Scripts\mousereach-watch.exe"
    "%USERPROFILE%\anaconda3\envs\mousereach\Scripts\mousereach-watch.exe"
) do (
    if exist %%P if not defined WATCH_EXE set "WATCH_EXE=%%~P"
)

if not defined WATCH_EXE (
    echo ERROR: Could not find mousereach-watch.exe
    echo Searched: %%MOUSEREACH_ENV%%\Scripts and the default conda env locations
    echo Run 'pip install -e .' in the MouseReach repo to install.
    pause
    exit /b 1
)

REM Derive the Scripts directory from the exe path
for %%F in ("%WATCH_EXE%") do set "SCRIPTS_DIR=%%~dpF"
set "STATUS_EXE=%SCRIPTS_DIR%mousereach-watch-status.exe"

echo Found MouseReach at: %SCRIPTS_DIR%

REM Wait for the shared pipeline folder (nas_root in ~/.mousereach/config.json).
REM Set MOUSEREACH_NAS_ROOT, MOUSEREACH_PROCESSING_ROOT and MOUSEREACH_ENV as user environment
REM variables on each machine (they mirror ~/.mousereach/config.json).
if defined MOUSEREACH_NAS_ROOT (
    echo Waiting for the NAS folder %MOUSEREACH_NAS_ROOT% ...
    :wait_nas
    if not exist "%MOUSEREACH_NAS_ROOT%" (
        timeout /t 5 /nobreak >nul
        goto wait_nas
    )
    echo NAS folder found.
)

REM Launch the watcher daemon in its own window
start "MouseReach Watcher" cmd /k "%WATCH_EXE%"

REM Give the watcher a moment to initialize and create the log file
timeout /t 10 /nobreak >nul

REM Find the watcher log file (<processing root>\watcher_logs\watcher.log). The roots come
REM from the environment (MOUSEREACH_PROCESSING_ROOT = this machine's local pipeline folder,
REM MOUSEREACH_NAS_ROOT = the shared one) -- no drive letters live in this script.
set "LOG_FILE="
for %%D in (
    "%MOUSEREACH_PROCESSING_ROOT%\watcher_logs\watcher.log"
    "%MOUSEREACH_NAS_ROOT%\watcher_logs\watcher.log"
) do (
    if exist %%D if not defined LOG_FILE set "LOG_FILE=%%~D"
)

if defined LOG_FILE (
    REM Launch a live log tail window
    start "MouseReach Log" powershell -ExecutionPolicy Bypass -NoExit -Command "Get-Content '%LOG_FILE%' -Tail 50 -Wait"
) else (
    echo WARNING: Could not find watcher.log - skipping log window
)

exit
