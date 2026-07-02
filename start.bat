@echo off
cd /d "%~dp0"
echo Starting ChatNest server...
echo You can access it at:
echo   http://localhost:8787
echo   http://192.168.1.105:8787  (from other devices on LAN)
echo.
echo Login password: chatnest123
echo.
".venv\Scripts\python.exe" run_server.py
