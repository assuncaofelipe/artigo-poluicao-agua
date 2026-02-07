#!/usr/bin/env python3
import argparse
import csv
import re
import time

import serial


LINE_RE = re.compile(r"Vout:\s*([0-9.]+)\s*V\s*\|\s*I:\s*([\-0-9.]+)\s*A")


def parse_line(line: str):
    m = LINE_RE.search(line)
    if not m:
        return None
    vout = float(m.group(1))
    current_a = float(m.group(2))
    return vout, current_a


def main():
    ap = argparse.ArgumentParser(description="Loga corrente/voltagem via serial (Arduino+ACS712).")
    ap.add_argument("--port", required=True, help="Ex.: /dev/ttyUSB0 ou /dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=9600)
    ap.add_argument("--out", default="power_samples.csv")
    ap.add_argument("--voltage", type=float, default=5.0, help="Tensão fixa da Raspberry (V)")
    ap.add_argument("--no-reset", action="store_true", help="Evita reset do Arduino (DTR/RTS baixos)")
    ap.add_argument("--skip-seconds", type=float, default=0.0, help="Ignora leituras nos primeiros N segundos")
    ap.add_argument("--flush-seconds", type=float, default=2.0)
    args = ap.parse_args()

    ser = serial.Serial(args.port, args.baud, timeout=1, dsrdtr=False)
    if args.no_reset:
        try:
            ser.setDTR(False)
            ser.setRTS(False)
        except Exception:
            pass
    time.sleep(2.0)  # aguarda boot do Arduino (se houver)

    header = ["timestamp", "current_mA", "voltage_V"]
    last_flush = time.time()
    start_time = time.time()

    with open(args.out, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(header)

        print(f"Logging em {args.out}. Ctrl+C para parar.")
        try:
            while True:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                parsed = parse_line(line)
                if not parsed:
                    continue
                vout, current_a = parsed
                current_mA = abs(current_a) * 1000.0
                if args.skip_seconds > 0 and (time.time() - start_time) < args.skip_seconds:
                    continue
                writer.writerow([time.time(), f"{current_mA:.6f}", f"{args.voltage:.6f}"])

                if time.time() - last_flush >= args.flush_seconds:
                    f.flush()
                    last_flush = time.time()
        except KeyboardInterrupt:
            pass
        finally:
            ser.close()


if __name__ == "__main__":
    main()
