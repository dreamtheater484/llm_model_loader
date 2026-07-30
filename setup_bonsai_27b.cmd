@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup_bonsai_27b.ps1" %*
exit /b %ERRORLEVEL%
