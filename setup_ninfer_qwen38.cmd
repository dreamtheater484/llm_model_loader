@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup_ninfer_qwen38.ps1" %*
exit /b %ERRORLEVEL%