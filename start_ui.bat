@echo off
REM Tear Film Analysis UI Launcher
REM Double-click this file to start the Streamlit UI

echo ============================================================
echo Starting Tear Film Analysis UI...
echo ============================================================
echo.
echo Browser will open automatically at http://localhost:8501
echo.
echo To stop the server, close this window or press Ctrl+C
echo ============================================================
echo.

cd /d "%~dp0"
python -m streamlit run tear_film_ui.py

pause
