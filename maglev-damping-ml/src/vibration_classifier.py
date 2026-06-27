"""
Pathway 2: Vibration Classifier

Looks at a rolling window of displacement readings from the HC-SR04 and figures
out what kind of disturbance is happening. We train it entirely on synthetic
waveforms, so it needs no hardware, and the exact same model runs on live sensor
data later.

The four classes line up with the maglev vibration problem from our paper:
  0 = no_disturbance     (just flat baseline + sensor noise)
  1 = low_freq_sway      (the lateral swaying maglev systems are prone to)
  2 = high_freq_vibration
  3 = impulse_shock      (a sudden hit or step)

How it works: take a window of displacement -> pull out time and frequency
features -> feed them to a random forest -> get a label.
"""

import numpy as np
from scipy.fft import rfft, rfftfreq
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# --- Config tied to the real rig ---
FS = 50.0          # HCSR-04 practical sample rate (Hz). Adjust to your loop rate.
WINDOW = 128       # samples per classification window (~2.5 s at 50 Hz)
BASELINE_MM = 32.0 # paper's effective sensing midpoint (20-45 mm range)
NOISE_MM = 0.4     # sensor noise floor

rng = np.random.default_rng(42)


def make_window(kind):
    """Generate one synthetic displacement window (in mm) for a given class."""
    t = np.arange(WINDOW) / FS
    base = BASELINE_MM + rng.normal(0, NOISE_MM, WINDOW)

    if kind == 0:  # no disturbance
        return base

    if kind == 1:  # low-frequency sway: 0.3-1.5 Hz, larger amplitude
        f = rng.uniform(0.3, 1.5)
        amp = rng.uniform(2.0, 5.0)
        phase = rng.uniform(0, 2 * np.pi)
        return base + amp * np.sin(2 * np.pi * f * t + phase)

    if kind == 2:  # high-frequency vibration: 5-20 Hz, smaller amplitude
        f = rng.uniform(5.0, 20.0)
        amp = rng.uniform(0.8, 2.5)
        phase = rng.uniform(0, 2 * np.pi)
        return base + amp * np.sin(2 * np.pi * f * t + phase)

    if kind == 3:  # impulse/shock: spike then exponential decay ring-down
        sig = base.copy()
        hit = rng.integers(WINDOW // 4, WINDOW // 2)
        amp = rng.uniform(4.0, 9.0)
        decay = rng.uniform(3.0, 8.0)
        f = rng.uniform(2.0, 8.0)
        tt = t[hit:] - t[hit]
        sig[hit:] += amp * np.exp(-decay * tt) * np.cos(2 * np.pi * f * tt)
        return sig

    raise ValueError(kind)


def extract_features(window):
    """Time-domain + frequency-domain features from one displacement window."""
    w = window - np.mean(window)          # remove DC (baseline distance)
    # Time domain
    std = np.std(w)
    rng_pp = np.ptp(w)                     # peak-to-peak
    abs_max = np.max(np.abs(w))
    rms = np.sqrt(np.mean(w ** 2))
    # crude "spikiness": kurtosis-like ratio, flags impulses
    spike = abs_max / (rms + 1e-9)
    # Frequency domain
    spec = np.abs(rfft(w))
    freqs = rfftfreq(WINDOW, 1 / FS)
    dom_freq = freqs[np.argmax(spec)]      # dominant frequency
    spec_energy = np.sum(spec)
    # fraction of energy below 2 Hz (sway band) vs above (vibration band)
    low_band = np.sum(spec[freqs < 2.0]) / (spec_energy + 1e-9)
    high_band = np.sum(spec[freqs >= 2.0]) / (spec_energy + 1e-9)
    return [std, rng_pp, abs_max, rms, spike, dom_freq, low_band, high_band]


def build_dataset(n_per_class=600):
    X, y = [], []
    for kind in range(4):
        for _ in range(n_per_class):
            X.append(extract_features(make_window(kind)))
            y.append(kind)
    return np.array(X), np.array(y)


if __name__ == "__main__":
    print("Generating synthetic dataset...")
    X, y = build_dataset()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)

    clf = RandomForestClassifier(n_estimators=120, max_depth=10, random_state=0)
    clf.fit(Xtr, ytr)

    names = ["no_disturbance", "low_freq_sway", "high_freq_vibration", "impulse_shock"]
    pred = clf.predict(Xte)
    print("\n=== Classification report (held-out synthetic) ===")
    print(classification_report(yte, pred, target_names=names))
    print("Confusion matrix:")
    print(confusion_matrix(yte, pred))

    feat_names = ["std", "ptp", "abs_max", "rms", "spike", "dom_freq", "low_band", "high_band"]
    print("\nFeature importances:")
    for n, imp in sorted(zip(feat_names, clf.feature_importances_), key=lambda p: -p[1]):
        print(f"  {n:10s} {imp:.3f}")

    joblib.dump(clf, "../models/vibration_clf.joblib")
    print("\nSaved model -> vibration_clf.joblib")
