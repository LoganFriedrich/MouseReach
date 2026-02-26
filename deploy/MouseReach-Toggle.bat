@echo off
REM ============================================================
REM MouseReach Toggle - Pause/Resume watcher processing
REM ============================================================
REM Double-click to toggle between filming mode (paused) and
REM processing mode (active). Useful during recording sessions.
REM
REM Install: copy this file to your Desktop
REM ============================================================

title MouseReach Toggle

REM --- Auto-detect the mousereach-watch-toggle executable ---
set "TOGGLE_EXE="
for %%P in (
    "A:\envs\MouseReach\Scripts\mousereach-watch-toggle.exe"
    "Y:\envs\MouseReach\Scripts\mousereach-watch-toggle.exe"
    "C:\envs\MouseReach\Scripts\mousereach-watch-toggle.exe"
) do (
    if exist %%P if not defined TOGGLE_EXE set "TOGGLE_EXE=%%~P"
)

if not defined TOGGLE_EXE (
    echo ERROR: Could not find mousereach-watch-toggle.exe
    echo Searched: A:\envs, Y:\envs, C:\envs
    echo Run 'pip install -e .' in the MouseReach repo to install.
    pause
    exit /b 1
)

"%TOGGLE_EXE%"
echo.
pause
