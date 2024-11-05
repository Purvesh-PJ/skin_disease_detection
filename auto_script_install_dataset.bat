@echo off

REM Specify the dataset owner, dataset name, and set download path to current directory with a "data" folder

set "DATASET_OWNER=kmader"
set "DATASET_NAME=skin-cancer-mnist-ham10000"
set "DOWNLOAD_PATH=%~dp0data"  REM %~dp0 gives the directory of the script

REM Create the 'data' folder within the script's directory if it doesn't exist

if not exist "%DOWNLOAD_PATH%" (
    mkdir "%DOWNLOAD_PATH%"
)

REM Download the Kaggle dataset to the specified folder

echo Downloading dataset...
kaggle datasets download -d %DATASET_OWNER%/%DATASET_NAME% -p "%DOWNLOAD_PATH%"

REM Unzip the dataset in the specified folder

echo Unzipping dataset...
powershell -Command "Expand-Archive -Path '%DOWNLOAD_PATH%\%DATASET_NAME%.zip' -DestinationPath '%DOWNLOAD_PATH%'"

REM Optional: Delete the zip file after extraction

del "%DOWNLOAD_PATH%\%DATASET_NAME%.zip"

echo Download and extraction complete!
pause
