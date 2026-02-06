@echo off
setlocal

echo ============================================
echo   Racecar AI - Windows Installer
echo ============================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python not found. Downloading Python installer...
    echo.

    :: Download Python installer using PowerShell
    powershell -Command "& { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.12.8/python-3.12.8-amd64.exe' -OutFile '%TEMP%\python_installer.exe' }"

    if not exist "%TEMP%\python_installer.exe" (
        echo ERROR: Failed to download Python installer.
        echo Please install Python manually from https://www.python.org/downloads/
        pause
        exit /b 1
    )

    echo Installing Python 3.12.8...
    echo This may take a few minutes and may request admin permissions.
    echo.

    :: Install Python silently with PATH option enabled
    "%TEMP%\python_installer.exe" /passive InstallAllUsers=0 PrependPath=1 Include_pip=1 Include_launcher=1

    if %errorlevel% neq 0 (
        echo ERROR: Python installation failed.
        echo Please install Python manually from https://www.python.org/downloads/
        echo Make sure to check "Add Python to PATH" during installation.
        pause
        exit /b 1
    )

    :: Clean up installer
    del "%TEMP%\python_installer.exe" >nul 2>&1

    echo Python installed successfully.
    echo.
    echo IMPORTANT: Please close this window and run this script again
    echo so that the new PATH takes effect.
    echo.
    pause
    exit /b 0
)

:: Show detected Python version
echo Found Python:
python --version
echo.

:: Check if pip is available
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo pip not found. Installing pip...
    python -m ensurepip --upgrade
)

:: Install dependencies
echo Installing dependencies...
echo.
pip install -r "%~dp0requirements.txt"

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to install dependencies.
    echo Trying with --user flag...
    pip install --user -r "%~dp0requirements.txt"
)

echo.
echo ============================================
echo   Installation complete!
echo ============================================
echo.
echo Starting Racecar AI...
echo.

:: Run the game from the script's directory
cd /d "%~dp0"
python main.py

pause
