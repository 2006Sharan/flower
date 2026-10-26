@echo off
title FloraVision AI - 1-Click GitHub & Cloud Publisher
color 0B

echo ======================================================================
echo           FLORAVISION AI - CLOUD DEPLOYMENT WIZARD
echo ======================================================================
echo.
echo Your GitHub display name is "Sharan EK".
echo Note: GitHub usernames cannot have spaces (e.g. sharanek or sharan-ek).
echo.
set /p GH_USER="Please enter your exact GitHub username (handle): "

if "%GH_USER%"=="" (
    echo [ERROR] Username cannot be empty!
    pause
    exit /b 1
)

echo.
echo [1/3] Setting up Git remote for https://github.com/%GH_USER%/flower.git ...
git remote remove origin >nul 2>nul
git remote add origin https://github.com/%GH_USER%/flower.git
git branch -M main

echo.
echo [2/3] Opening GitHub in your browser to create the repository if needed...
start "" "https://github.com/new?name=flower&description=FloraVision+AI+Flower+Species+Classification+and+YOLO+Suite"

echo.
echo Please click "Create repository" in the browser window that just opened,
echo then press any key below to push your code!
pause

echo.
echo [3/3] Pushing code to GitHub...
git push -u origin main

if %errorlevel% equ 0 (
    echo.
    echo ======================================================================
    echo [SUCCESS] Code uploaded to GitHub successfully!
    echo ======================================================================
    echo.
    echo Now launching Render.com to generate your 24/7 public link...
    start "" "https://dashboard.render.com/select-repo?type=web"
    echo.
    echo In Render:
    echo 1. Select the "flower" repository.
    echo 2. Click "Create Web Service".
    echo 3. You will receive your permanent link: https://floravision-ai.onrender.com
) else (
    echo.
    echo [ERROR] Push failed. Please check that the repository was created and your credentials are correct.
)

pause
