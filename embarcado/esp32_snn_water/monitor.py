import serial
import sys
import time

try:
    ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
    print('========== MONITOR SERIAL SNN ==========')
    print('Aguardando boot do ESP32...\n')
    
    start = time.time()
    boot_done = False
    
    while time.time() - start < 20:
        try:
            line = ser.readline()
            if line:
                text = line.decode('utf-8', errors='replace')
                print(text, end='')
                if 'SNN' in text and 'Started' in text:
                    boot_done = True
        except:
            pass
    
    ser.close()
    print('\n========== FIM DO MONITOR ==========')
except Exception as e:
    print(f'Erro: {e}')
