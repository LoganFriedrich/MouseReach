@echo off
REM ============================================================
REM MouseReach Watcher - Auto-launch on login
REM ============================================================
REM Starts the watcher daemon and a live log in separate windows; both close
REM when the watcher stops. Close the watcher window (or Ctrl+C) to stop processing.
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

REM Launch the watcher daemon in its own window. The window closes when the
REM watcher stops cleanly (watcher_stop.flag, or the Stop button); if the watcher
REM fails, it stays open on "Press any key" so the error can be read.
REM WHY not "cmd /k": that kept every old window open after its watcher ended,
REM so each restart left one more behind (2026-10-22).
start "MouseReach Watcher" cmd /c ""%WATCH_EXE%" || pause"

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
    REM A live log tail window that ENDS WITH THE WATCHER. WHY: the old tail
    REM (Get-Content -Wait in a -NoExit window) never ended, so every restart left
    REM one more window holding watcher.log open; accumulated ones blocked folder
    REM renames on the lab PC (2026-08-30). The tail runs as a background job and
    REM the window closes once no mousereach-watch process is left.
    start "MouseReach Log" powershell -ExecutionPolicy Bypass -Command "$j = Start-Job { Get-Content -LiteralPath '%LOG_FILE%' -Tail 50 -Wait }; while (Get-Process -Name 'mousereach-watch' -ErrorAction SilentlyContinue) { Receive-Job $j; Start-Sleep -Seconds 1 }; Receive-Job $j; Stop-Job $j; Remove-Job $j"
) else (
    echo WARNING: Could not find watcher.log - skipping log window
)

exit
