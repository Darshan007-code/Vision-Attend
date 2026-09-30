@echo off
title VisionAttend - Biometric Attendance System
echo ======================================================================
echo   VISIONATTEND: AUTOMATED FACE RECOGNITION ATTENDANCE SYSTEM
echo   Department of Computer Science ^& Engineering - Enterprise Edition
echo ======================================================================
echo.
echo [*] Initializing database and starting VisionAttend Web Portal...
echo [*] Launching browser at http://localhost:5000 in 3 seconds...
echo.

start "" "http://localhost:5000"
python app.py

pause
