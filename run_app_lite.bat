@echo off
echo =============================================
echo Audio Deepfake Detector - PROFESSIONAL LITE
echo =============================================
echo.
echo Starting professional Gradio app...
echo.

cd /d "%~dp0"
call v\Scripts\activate.bat
python app_gradio_lite.py

pause
