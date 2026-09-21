@echo off
cd /d "%~dp0"
echo Opening Jupyter. Your browser will open in a few seconds.
echo Close this window when you are done.
".venv\Scripts\jupyter.exe" lab "notebooks\01_load.ipynb" --ip=127.0.0.1 --port=8888
