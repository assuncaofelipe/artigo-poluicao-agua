@echo off
REM setup_esp32.bat - Helper para setup de MicroPython na ESP32 (Windows)
REM Uso: setup_esp32.bat COM4

setlocal enabledelayedexpansion

set PORT=%1
if "%PORT%"=="" (
    set PORT=COM4
)

echo ==========================================
echo Setup ESP32 MicroPython - SNN
echo ==========================================
echo Porta: %PORT%
echo.

REM 1. Instala ferramentas no PC
echo [1/3] Instalando ferramentas no PC...
python -m pip install --upgrade mpremote esptool >nul 2>&1
if errorlevel 1 (
    echo   ^(aviso: pip pode ter falhado^)
)

REM 2. Verifica conexão
echo.
echo [2/3] Verificando conexao com ESP32...
mpremote connect %PORT% help >nul 2>&1
if errorlevel 1 (
    echo   ERRO: Nao consegui conectar a %PORT%
    echo.
    echo   Certifique-se de que:
    echo   - A ESP32 esta conectada ao PC
    echo   - O firmware MicroPython esta instalado
    echo   - Nenhum outro programa esta usando a porta
    pause
    exit /b 1
)
echo   OK: ESP32 detectada em %PORT%

REM 3. Faz upload
echo.
echo [3/3] Fazendo upload dos arquivos...
mpremote connect %PORT% fs cp main.py :main.py
if errorlevel 1 (
    echo   ERRO ao fazer upload de main.py
    pause
    exit /b 1
)

mpremote connect %PORT% fs cp model_weights.json :model_weights.json
if errorlevel 1 (
    echo   ERRO ao fazer upload de model_weights.json
    pause
    exit /b 1
)
echo   OK: Upload completo

echo.
echo ==========================================
echo OK: Setup concluido!
echo ==========================================
echo.
echo Proximas etapas:
echo 1. Conecte a ESP32:
echo    mpremote connect %PORT%
echo.
echo 2. No REPL da ESP32, rode:
echo    ^>^>^> import main
echo    ^>^>^> main.benchmark^(^)
echo.
pause
