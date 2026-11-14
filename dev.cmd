@echo off
where pwsh.exe >nul 2>nul
if errorlevel 1 (
    powershell.exe -NoLogo -NoProfile -File "%~dp0dev.ps1" %*
) else (
    pwsh.exe -NoLogo -NoProfile -File "%~dp0dev.ps1" %*
)
exit /b %errorlevel%
