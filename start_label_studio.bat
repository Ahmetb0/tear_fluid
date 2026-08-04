@echo off
REM Shortcut: delegates to the global launcher (works even if this project is deleted)
call "%APPDATA%\Python\Python311\Scripts\label-studio-start.bat" %*
pause
