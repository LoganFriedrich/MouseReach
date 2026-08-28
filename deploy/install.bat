@echo off
REM ============================================================
REM MouseReach Deploy - Pull latest + install startup scripts
REM ============================================================
REM Run this on any lab PC. It will:
REM   1. Detect which machine this is
REM   2. Pull the latest code from git
REM   3. Install the appropriate startup scripts
REM
REM Usage: just double-click this file, or run from any terminal.
REM ============================================================

title MouseReach Deploy
echo.
echo ==============================================================
echo  MouseReach Deploy
echo ==============================================================
echo.

REM --- Identify this machine ---
set "MACHINE_ROLE=Unknown"
set "NEEDS_WATCHER=0"
set "NEEDS_TOGGLE=0"

for /f "tokens=*" %%H in ('hostname') do set "HOSTNAME=%%H"
echo Machine: %HOSTNAME%

if /i "%HOSTNAME%"=="EXAMPLE-HOST-2" (
    set "MACHINE_ROLE=NAS / DLC PC"
    set "NEEDS_WATCHER=1"
    set "NEEDS_TOGGLE=0"
    echo Role:    NAS / DLC PC ^(primary DLC processing^)
)
if /i "%HOSTNAME%"=="EXAMPLE-HOST-1" (
    set "MACHINE_ROLE=Vid&DLC1PC"
    set "NEEDS_WATCHER=1"
    set "NEEDS_TOGGLE=1"
    echo Role:    Vid^&DLC1PC ^(filming + DLC^)
)
if /i "%HOSTNAME%"=="EXAMPLE-HOST-3" (
    set "MACHINE_ROLE=Vid&DLC2PC"
    set "NEEDS_WATCHER=1"
    set "NEEDS_TOGGLE=1"
    echo Role:    Vid^&DLC2PC ^(filming + DLC^)
)
if "%MACHINE_ROLE%"=="Unknown" (
    REM Processing Server: the machine that hosts the shared pipeline folder locally
    if defined MOUSEREACH_NAS_ROOT if exist "%MOUSEREACH_NAS_ROOT%" (
        set "MACHINE_ROLE=Processing Server"
        set "NEEDS_WATCHER=1"
        set "NEEDS_TOGGLE=0"
        echo Role:    Processing Server
    )
)

if "%MACHINE_ROLE%"=="Unknown" (
    echo.
    echo WARNING: Could not identify this machine.
    echo   Hostname: %HOSTNAME%
    echo   Expected: EXAMPLE-HOST-2, EXAMPLE-HOST-1, EXAMPLE-HOST-3,
    echo             or a machine with G:\ and Y:\ drives.
    echo.
    echo   No scripts will be installed. Run mousereach-setup --set-role
    echo   to configure this machine first.
    pause
    exit /b 1
)

echo.

REM --- Find the repo ---
set "REPO_DIR="
for %%R in (
    "A:\Behavior\MouseReach"
    "Y:\Behavior\MouseReach"
    "C:\Behavior\MouseReach"
) do (
    if exist "%%~R\.git" if not defined REPO_DIR set "REPO_DIR=%%~R"
)

if not defined REPO_DIR (
    echo ERROR: Could not find MouseReach git repo.
    echo Searched: A:\Behavior\MouseReach, Y:\Behavior\MouseReach, C:\Behavior\MouseReach
    pause
    exit /b 1
)

echo Repo:    %REPO_DIR%

REM --- Pull latest master ---
echo.
echo Pulling latest master...
pushd "%REPO_DIR%"
git fetch origin
git checkout master
git pull origin master
if errorlevel 1 (
    echo.
    echo WARNING: git pull failed. Continuing with current version.
)
popd

REM --- Install pip package if needed ---
set "WATCH_EXE="
for %%P in (
    "A:\envs\MouseReach\Scripts\mousereach-watch.exe"
    "Y:\envs\MouseReach\Scripts\mousereach-watch.exe"
    "C:\envs\MouseReach\Scripts\mousereach-watch.exe"
) do (
    if exist %%P if not defined WATCH_EXE set "WATCH_EXE=%%~P"
)

if not defined WATCH_EXE (
    echo.
    echo WARNING: mousereach-watch.exe not found.
    echo You may need to run: pip install -e "%REPO_DIR%"
    echo Continuing with script installation anyway...
)

REM --- Install startup script ---
set "STARTUP_DIR=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"
set "DEPLOY_DIR=%REPO_DIR%\deploy"

if "%NEEDS_WATCHER%"=="1" (
    echo.
    echo Installing watcher startup script...
    copy /y "%DEPLOY_DIR%\MouseReach-Watcher-Startup.bat" "%STARTUP_DIR%\MouseReach-Watcher.bat" >nul
    if errorlevel 1 (
        echo   FAILED to copy startup script.
    ) else (
        echo   Installed: %STARTUP_DIR%\MouseReach-Watcher.bat
    )
)

REM --- Install toggle shortcut (filming PCs only) ---
if "%NEEDS_TOGGLE%"=="1" (
    echo.
    echo Installing toggle shortcut to Desktop...
    copy /y "%DEPLOY_DIR%\MouseReach-Toggle.bat" "%USERPROFILE%\Desktop\MouseReach Toggle.bat" >nul
    if errorlevel 1 (
        echo   FAILED to copy toggle shortcut.
    ) else (
        echo   Installed: %USERPROFILE%\Desktop\MouseReach Toggle.bat
    )
)

REM --- Summary ---
echo.
echo ==============================================================
echo  Deploy complete!  (%MACHINE_ROLE%)
echo ==============================================================
echo.
if "%NEEDS_WATCHER%"=="1" (
    echo  [x] Watcher auto-starts on login
)
if "%NEEDS_TOGGLE%"=="1" (
    echo  [x] Toggle shortcut on Desktop
)
echo.
echo  Next login will auto-launch the watcher and status monitor.
echo  Or start now:  mousereach-watch
echo  Check status:  mousereach-watch-status
echo.

pause
