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
    "A:\envs\MouseReach\Scripts\mousereach-watch.exe"
    "Y:\envs\MouseReach\Scripts\mousereach-watch.exe"
    "C:\envs\MouseReach\Scripts\mousereach-watch.exe"
) do (
    if exist %%P if not defined WATCH_EXE set "WATCH_EXE=%%~P"
)

if not defined WATCH_EXE (
    echo ERROR: Could not find mousereach-watch.exe
    echo Searched: A:\envs, Y:\envs, C:\envs
    echo Run 'pip install -e .' in the MouseReach repo to install.
    pause
    exit /b 1
)

REM Derive the Scripts directory from the exe path
for %%F in ("%WATCH_EXE%") do set "SCRIPTS_DIR=%%~dpF"
set "STATUS_EXE=%SCRIPTS_DIR%mousereach-watch-status.exe"

echo Found MouseReach at: %SCRIPTS_DIR%

REM Wait for network drives to be available (Y: is NAS)
echo Waiting for NAS drive (Y:\)...
:wait_nas
if not exist "Y:\2_Connectome" (
    timeout /t 5 /nobreak >nul
    goto wait_nas
)
echo NAS drive found.

REM Launch the watcher daemon in its own window
start "MouseReach Watcher" cmd /k "%WATCH_EXE%"

REM Give the watcher a moment to initialize
timeout /t 10 /nobreak >nul

REM Launch a status monitor that refreshes every 60 seconds
start "MouseReach Status" cmd /k "title MouseReach Status && :loop && cls && "%STATUS_EXE%" && echo. && echo --- Refreshing in 60s (Ctrl+C to stop) --- && timeout /t 60 /nobreak >nul && goto loop"

exit
