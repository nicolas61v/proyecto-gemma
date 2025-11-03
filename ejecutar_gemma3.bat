@echo off
echo ========================================
echo    CHAT GEMMA 3 270M - Universidad EAFIT
echo ========================================
echo.
echo Iniciando aplicacion con el modelo CORRECTO...
echo.
echo GEMMA 3 270M - Ventajas:
echo - Solo 270M parametros (ultra rapido!)
echo - Disenado para fine-tuning
echo - Descarga: ~241MB (vs 5GB del 2B)
echo - Perfecto para tu proyecto!
echo.
echo Primera ejecucion descargara el modelo (~241MB)
echo.
echo Presiona Ctrl+C para detener
echo.
echo ========================================
echo.

call venv\Scripts\activate.bat
python gemma3_270m_chat.py

pause
