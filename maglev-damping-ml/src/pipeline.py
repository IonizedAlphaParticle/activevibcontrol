"""
Integrated Control Pipeline: sense -> classify -> adapt

This is where the two models come together, and it's the heart of the demo. One
loop does four things:

  1. SENSE    - grab a window of displacement readings (from the HC-SR04, or
                synthetic data here)
  2. CLASSIFY - figure out what kind of disturbance it is: nothing, sway,
                vibration, or a shock
  3. ADAPT    - pick the right Kp/Kd for the current duty cycle, then nudge the
                response based on what kind of disturbance we're dealing with
  4. ACTUATE  - on real hardware, send the PWM command to the electromagnet

It runs entirely on synthetic data, so youu can demo the whole thing with nothing
plugged in. When you're ready to go live, swap synthetic_stream() for the serial
reader.
"""

import numpy as np
import joblib
import os

HERE = os.path.dirname(__file__)
MODELS = os.path.join(HERE, "..", "models")

clf = joblib.load(os.path.join(MODELS, "vibration_clf.joblib"))
gain_model = joblib.load(os.path.join(MODELS, "gain_model.joblib"))

# import feature/sim helpers from the two pathway modules
import sys
sys.path.insert(0, HERE)
from vibration_classifier import extract_features, make_window, WINDOW, FS
from gain_model import pwm_to_field, control_authority

CLASS_NAMES = ["no_disturbance", "low_freq_sway", "high_freq_vibration", "impulse_shock"]

# Per-class control bias: how aggressively to respond to each disturbance type.
# Sway is low-freq and persistent -> favor proportional. Shock is transient ->
# favor derivative. These are interpretable multipliers on the base gains.
CLASS_BIAS = {
    "no_disturbance":      (0.5, 0.5),   # back off, nothing to fight
    "low_freq_sway":       (1.2, 0.9),
    "high_freq_vibration": (0.9, 1.2),
    "impulse_shock":       (1.0, 1.4),   # lean on derivative to catch the spike
}


def step(window, ocr):
    """One full sense->classify->adapt cycle. Returns a decision dict."""
    # CLASSIFY
    feats = extract_features(window)
    cls_idx = int(clf.predict([feats])[0])
    cls = CLASS_NAMES[cls_idx]

    # ADAPT: base gains for this operating point
    B = pwm_to_field(ocr)
    auth = control_authority(B)
    base_kp, base_kd = gain_model.predict([[ocr, B, auth]])[0]

    # apply class-specific bias
    bp, bd = CLASS_BIAS[cls]
    kp, kd = base_kp * bp, base_kd * bd

    return {
        "class": cls,
        "field_G": round(float(B), 1),
        "base_gains": (round(float(base_kp), 1), round(float(base_kd), 1)),
        "final_gains": (round(float(kp), 1), round(float(kd), 1)),
    }


def synthetic_stream(n=8):
    """Yield (window, ocr) pairs cycling through disturbance types for demo."""
    rng = np.random.default_rng(7)
    for i in range(n):
        kind = i % 4
        ocr = int(rng.uniform(500, 950))
        yield make_window(kind), ocr, CLASS_NAMES[kind]


if __name__ == "__main__":
    print("=== Integrated sense -> classify -> adapt demo (synthetic) ===\n")
    correct = 0
    total = 0
    for window, ocr, true_cls in synthetic_stream(12):
        d = step(window, ocr)
        ok = "OK" if d["class"] == true_cls else "XX"
        if d["class"] == true_cls:
            correct += 1
        total += 1
        print(f"[{ok}] true={true_cls:20s} pred={d['class']:20s} "
              f"OCR={ocr:3d} B={d['field_G']:5.1f}G  "
              f"gains -> Kp={d['final_gains'][0]:5.1f} Kd={d['final_gains'][1]:5.1f}")
    print(f"\nLive-classification accuracy on demo stream: {correct}/{total}")
