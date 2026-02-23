import serial
import csv
import time

PORTA = 'COM5'
BAUDRATE = 9600  # Ajuste conforme necessário
TIMEOUT = 1     # Timeout em segundos
CSV_PATH = 'serial_capture.csv'

def main():
    try:
        with serial.Serial(PORTA, BAUDRATE, timeout=TIMEOUT) as ser, open(CSV_PATH, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            print(f'Capturando dados da porta {PORTA} e salvando em {CSV_PATH}. Pressione Ctrl+C para sair.')
            while True:
                if ser.in_waiting > 0:
                    linha = ser.readline().decode('utf-8', errors='replace').strip()
                    if linha:
                        # Tenta separar por vírgula, ponto e vírgula ou espaço
                        if ',' in linha:
                            row = linha.split(',')
                        elif ';' in linha:
                            row = linha.split(';')
                        else:
                            row = linha.split()
                        writer.writerow(row)
                        print(row)
    except serial.SerialException as e:
        print(f'Erro ao acessar a porta serial: {e}')
    except KeyboardInterrupt:
        print('\nLeitura encerrada pelo usuário.')

if __name__ == '__main__':
    main()
