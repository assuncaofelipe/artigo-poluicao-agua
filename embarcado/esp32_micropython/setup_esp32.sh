#!/usr/bin/env bash
# setup_esp32.sh - Helper para setup de MicroPython na ESP32
# Uso: bash setup_esp32.sh <COM_PORT>

set -e

PORT=${1:-COM4}

echo "=========================================="
echo "Setup ESP32 MicroPython - SNN"
echo "=========================================="
echo "Porta: $PORT"
echo ""

# 1. Instala ferramentas no PC
echo "[1/3] Instalando ferramentas no PC..."
pip install --upgrade mpremote esptool 2>/dev/null || echo "  (pode falhar em alguns ambientes)"

# 2. Verifica se ESP32 está acessível
echo ""
echo "[2/3] Verificando conexão com ESP32..."
if ! mpremote connect "$PORT" help > /dev/null 2>&1; then
    echo "  ⚠️  AVISO: Não consegui conectar a $PORT"
    echo "  Certifique-se de que:"
    echo "  - A ESP32 está conectada ao PC"
    echo "  - O firmware MicroPython está instalado"
    echo "  - Nenhum outro programa está usando a porta"
    exit 1
fi
echo "  ✔ ESP32 detectada em $PORT"

# 3. Faz upload dos arquivos
echo ""
echo "[3/3] Fazendo upload dos arquivos..."
mpremote connect "$PORT" fs cp main.py :main.py
mpremote connect "$PORT" fs cp model_weights.json :model_weights.json
echo "  ✔ Upload completo"

echo ""
echo "=========================================="
echo "✅ Setup concluído!"
echo "=========================================="
echo ""
echo "Próximas etapas:"
echo "1. Conecte à ESP32:"
echo "   mpremote connect $PORT"
echo ""
echo "2. No REPL, rode:"
echo "   >>> import main"
echo "   >>> main.benchmark()"
echo ""
