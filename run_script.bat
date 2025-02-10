@echo off
echo Starting script execution at %date% %time%
cd /d "%~dp0"
call C:\Users\user\Documents\pinecone_update\venv\Scripts\activate.bat
IF %ERRORLEVEL% NEQ 0 (
    echo Failed to activate virtual environment
    pause
    exit /b 1
)
python -c "import sys; print(sys.executable)"
python main.py
IF %ERRORLEVEL% NEQ 0 (
    echo Script failed with error code %ERRORLEVEL%
    pause
    exit /b 1
)
echo Script completed at %date% %time%
pause