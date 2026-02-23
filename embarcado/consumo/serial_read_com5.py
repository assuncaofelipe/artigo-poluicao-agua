import serial

# Configurações da porta serial
PORTA = 'COM5'
BAUDRATE = 9600  # Ajuste conforme necessário
TIMEOUT = 1     # Timeout em segundos

def main():
    try:
        with serial.Serial(PORTA, BAUDRATE, timeout=TIMEOUT) as ser:
            print(f'Conectado à porta {PORTA}. Pressione Ctrl+C para sair.')
            while True:
                if ser.in_waiting > 0:
                    linha = ser.readline().decode('utf-8', errors='replace').strip()
                    print(linha)
    except serial.SerialException as e:
        print(f'Erro ao acessar a porta serial: {e}')
    except KeyboardInterrupt:
        print('\nLeitura encerrada pelo usuário.')

if __name__ == '__main__':
    main()
