@echo off
title VisionAttend - Standalone Desktop Biometric Kiosk
echo ======================================================================
echo   VISIONATTEND: STANDALONE DESKTOP BIOMETRIC KIOSK (OPENCV MODE)
echo   Department of Computer Science ^& Engineering - Enterprise Edition
echo ======================================================================
echo.
echo [*] Launching Native OpenCV Window...
echo [*] Controls: [Q] Quit  [S] Switch Subject  [E] Enroll Face  [T] Train AI
echo.

python run_gui.py

pause
