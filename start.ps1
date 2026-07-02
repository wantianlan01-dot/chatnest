$env:AGENT_APP_ROOT = "C:\Users\34367\OneDrive\桌面\Codexx\chatnest"
Set-Location $env:AGENT_APP_ROOT
& ".venv\Scripts\python.exe" -m uvicorn app.main:app --host 0.0.0.0 --port 8787
