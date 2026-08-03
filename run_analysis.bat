@echo off
REM Quick Analysis Launcher
REM Double-click to run analysis with default settings

echo ============================================================
echo Tear Film Analysis - Quick Run
echo ============================================================
echo.
echo Running analysis with default parameters...
echo Video: AYDIN_MEHMET TUNAHAN_Right_2026_07_26-14_01_16.mkv
echo.
echo This will take approximately 15-20 seconds.
echo ============================================================
echo.

cd /d "%~dp0"
python tear_film_advanced.py

echo.
echo ============================================================
echo Analysis complete! Check tear_film_analysis_advanced.csv
echo ============================================================
echo.

pause
