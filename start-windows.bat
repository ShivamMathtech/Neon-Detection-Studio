@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  py -3.11 -m venv .venv
  if errorlevel 1 goto fail
)
.venv\Scripts\python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto fail
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto fail
.venv\Scripts\python.exe scripts\download_model.py
if errorlevel 1 goto fail
echo Open http://127.0.0.1:8765 in your browser.
.venv\Scripts\python.exe server.py
exit /b
:fail
echo Setup failed. Install Python 3.11 and check the error above.
pause
