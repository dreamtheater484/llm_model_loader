@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup_qwen38_nvfp4.ps1" %*
exit /b %ERRORLEVEL%
