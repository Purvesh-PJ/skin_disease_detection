@echo off
setlocal

REM Check for Node.js
echo Checking for Node.js...
node -v >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo Node.js is not installed. Installing Node.js...
    curl -o nodejs.msi https://nodejs.org/dist/v16.20.0/node-v16.20.0-x64.msi
    msiexec /i nodejs.msi /quiet /norestart
    del nodejs.msi
) ELSE (
    echo Node.js is already installed. Version: 
    node -v
)

REM Check for Python
echo Checking for Python...
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo Python is not installed. Installing Python...
    curl -o python.exe https://www.python.org/ftp/python/3.8.10/python-3.8.10-amd64.exe
    start /wait python.exe /quiet InstallAllUsers=1 PrependPath=1
    del python.exe
) ELSE (
    echo Python is already installed. Version: 
    python --version
)

REM Check for Git
echo Checking for Git...
git --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo Git is not installed. Installing Git...
    curl -o git.exe https://github.com/git-for-windows/git/releases/download/v2.39.1.windows.1/Git-2.39.1-64-bit.exe
    start /wait git.exe /silent /norestart
    del git.exe
) ELSE (
    echo Git is already installed. Version:
    git --version
)

REM Check for Virtualenv
echo Checking for Virtualenv...
pip show virtualenv >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo Virtualenv is not installed. Installing Virtualenv...
    python -m pip install --upgrade pip
    pip install virtualenv
) ELSE (
    echo Virtualenv is already installed.
)

echo All prerequisites are installed successfully!
endlocal
pause
