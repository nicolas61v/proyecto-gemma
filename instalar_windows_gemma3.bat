@echo off
echo ========================================
echo    INSTALACION GEMMA 3 270M
echo    Universidad EAFIT - 2025
echo ========================================
echo.
echo Modelo CORRECTO: Gemma 3 270M
echo - 270M parametros (ultra compacto)
echo - Entrenado con 6 trillones de tokens
echo - Disenado para fine-tuning rapido
echo - Perfecto para tu proyecto!
echo.

:: Verificar Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python no esta instalado!
    echo Por favor instala Python 3.8+ desde https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [OK] Python detectado
echo.

:: Crear entorno virtual
echo [1/4] Creando entorno virtual...
python -m venv venv
if errorlevel 1 (
    echo [ERROR] No se pudo crear el entorno virtual
    pause
    exit /b 1
)
echo [OK] Entorno virtual creado
echo.

:: Activar entorno virtual
echo [2/4] Activando entorno virtual...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] No se pudo activar el entorno virtual
    pause
    exit /b 1
)
echo [OK] Entorno activado
echo.

:: Actualizar pip
echo [3/4] Actualizando pip...
python -m pip install --upgrade pip --quiet
echo [OK] pip actualizado
echo.

:: Instalar dependencias
echo [4/4] Instalando dependencias (5-10 minutos)...
pip install -r requirements_gemma3.txt --quiet
if errorlevel 1 (
    echo [ERROR] Fallo la instalacion de dependencias
    pause
    exit /b 1
)
echo [OK] Dependencias instaladas
echo.

echo.
echo ========================================
echo    INSTALACION COMPLETADA!
echo ========================================
echo.
echo SIGUIENTE PASO:
echo 1. Configura tu token de Hugging Face:
echo    huggingface-cli login
echo.
echo 2. Ejecuta la aplicacion con:
echo    ejecutar_gemma3.bat
echo.
echo Para mas informacion, lee README_GEMMA3_270M.md
echo.
pause
