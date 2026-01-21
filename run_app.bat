@echo off
echo ========================================
echo  Audio Deepfake Detector - Starting...
echo ========================================
echo.

REM Activate virtual environment
echo [1/3] Activating virtual environment...
call v\Scripts\activate.bat

REM Check if activation was successful
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    echo Please ensure the virtual environment exists in the 'v' folder
    pause
    exit /b 1
)

echo [2/3] Checking Python and Gradio installation...
python --version
python -c "import gradio; print(f'Gradio version: {gradio.__version__}')" 2>nul
if errorlevel 1 (
    echo WARNING: Gradio not found. Installing requirements...
    pip install -r requirements.txt
)

echo.
echo [3/3] Starting Gradio application...
echo ----------------------------------------
echo.
python app_gradio.py

REM Keep window open if there's an error
if errorlevel 1 (
    echo.
    echo ========================================
    echo ERROR: Application crashed!
    echo ========================================
    pause
)
