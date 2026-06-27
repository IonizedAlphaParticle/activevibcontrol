import argparse
import collections
import numpy as np

from vibration_classifier import WINDOW
from pipeline import step, CLASS_NAMES


def run(port, baud=115200, ocr=750):
    try:
        import serial
    except ImportError:
        raise SystemExit("pyserial not installed. Run: pip install pyserial")

    ser = serial.Serial(port, baud, timeout=1)
    buf = collections.deque(maxlen=WINDOW)
    print(f"Listening on {port} @ {baud} baud. Ctrl-C to stop.\n")

    try:
        while True:
            line = ser.readline().decode(errors="ignore").strip()
            if not line:
                continue
            try:
                val = float(line)
            except ValueError:
                continue
            buf.append(val)
            if len(buf) == WINDOW:
                window = np.array(buf)
                d = step(window, ocr)
                print(f"\r{d['class']:20s} | field {d['field_G']:5.1f}G | "
                      f"Kp={d['final_gains'][0]:5.1f} Kd={d['final_gains'][1]:5.1f}",
                      end="", flush=True)
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        ser.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--ocr", type=int, default=750, help="current PWM operating point")
    args = ap.parse_args()
    run(args.port, args.baud, args.ocr)
