# Adaptive ML Control for Electromagnetic Vibration Damping

An ML layer on top of a PWM-controlled electromagnetic damping module for
maglev / hyperloop vibration suppression. Built for **vsHacks 2026**.

This project extends a physical proof-of-concept damping module (a PD-controlled
electromagnet that damps vibration via eddy currents in a conductive rail) with
two machine-learning components that address the system's core difficulty: the
relationship between control input and damping force is **strongly nonlinear**.

The hardware characterization that grounds this work (PWM-to-field curve,
spatial field decay, eddy-current force model, FEMM simulation) comes from our
companion research paper. The ML here is built directly on those measured
relationships, not on generic assumptions.

---

## The core problem

The module's measured PWM-to-field relationship is nonlinear:

```
B(OCR) = 0.0644 · e^(0.00706 · OCR) + 7.6024     (R² = 0.975)
```

and the eddy-current damping force scales with the **square** of the field
(`F ∝ B² ∝ I²`). This has two consequences a naive controller ignores:

1. Below ~45% duty cycle the electromagnet sits in a **dead zone** (field pinned
   near the ~7 G noise floor) where no gain choice produces useful damping.
2. In the active region, the *same* PD gains behave completely differently at
   different duty cycles: sluggish where the field is weak, twitchy and
   overshoot-prone where the field is strong.

A single fixed Kp/Kd pair therefore cannot be good everywhere. That is the gap
the ML fills.

---

## What the ML does

### Pathway 1 — Adaptive PD gain tuning (`src/gain_model.py`)
A regression model predicts the Kp/Kd that minimize settling time and overshoot
**as a function of the operating point** (duty cycle → field → control
authority). Training data is generated from the paper's own mass-spring-damper
simulation (`m·x'' + c·x' + k·x = 0`, with `c ∝ B²`), so the labels are grounded
in the measured physics.

**Result:** against fixed gains tuned at full field, the adapted gains cut
overshoot substantially across the low-and-mid field range:

| Operating point | Fixed-gain overshoot | ML-adapted overshoot |
|-----------------|----------------------|----------------------|
| OCR 500 (≈10 G) | 45%                  | 24%                  |
| OCR 650 (≈14 G) | 42%                  | 20%                  |
| OCR 800 (≈26 G) | 28%                  | 6%                   |
| OCR 950 (≈60 G) | 0% (tuned here)      | 0%                   |

![Fixed vs adapted](figures/fig_adaptive_vs_fixed.png)

### Pathway 2 — Real-time vibration classifier (`src/vibration_classifier.py`)
A RandomForest classifies the displacement stream into one of four disturbance
types that map to the maglev vibration problem:

- `no_disturbance`
- `low_freq_sway` (the lateral swaying maglev systems suffer from)
- `high_freq_vibration`
- `impulse_shock`

It uses time-domain features (RMS, peak-to-peak, spikiness) plus FFT-based
features (dominant frequency, low/high frequency-band energy split). Trained
entirely on synthetic waveforms so it needs **no hardware**, then runs
identically on live HCSR-04 data.

### Integration — sense → classify → adapt (`src/pipeline.py`)
The two combine into one loop: the classifier identifies *what* the disturbance
is, then the gain model picks gains for the operating point, biased by the
disturbance class (e.g. lean on the derivative term for transient shocks). This
is the demoable centerpiece and runs offline on synthetic data.

---

## Running it

```bash
pip install -r requirements.txt

# Train + evaluate the classifier
python src/vibration_classifier.py

# Train the gain model + print the fixed-vs-adapted comparison
python src/gain_model.py

# Regenerate the comparison figure
python src/make_figures.py

# Run the integrated sense->classify->adapt pipeline (no hardware needed)
python src/pipeline.py
```

### Going live with hardware (optional)
The HCSR-04 ultrasonic sensor runs off the Arduino's USB 5V — the 7.26 V
electromagnet supply is **not** required to demo the ML.

1. Flash `arduino/displacement_streamer.ino` to an Arduino Uno.
2. Stream into the pipeline:
   ```bash
   pip install pyserial
   python src/live_serial.py --port /dev/ttyACM0   # or COM3 on Windows
   ```

---

## Repository layout

```
maglev-damping-ml/
├── src/
│   ├── vibration_classifier.py   # Pathway 2: disturbance classifier
│   ├── gain_model.py             # Pathway 1: adaptive PD gains
│   ├── pipeline.py               # integrated sense->classify->adapt
│   ├── live_serial.py            # optional live HCSR-04 bridge
│   └── make_figures.py           # generates the comparison figure
├── models/                       # trained model artifacts (.joblib)
├── figures/                      # generated plots
├── arduino/
│   └── displacement_streamer.ino # streams displacement over USB serial
├── requirements.txt
└── LICENSE                       # MIT
```

---

## Honesty notes (what is and isn't validated)

- The gain-tuning benefit is demonstrated **in simulation**, using a plant model
  parameterized from measured field data. Hardware-in-the-loop validation
  (deploying adapted gains back to the physical rig) is the immediate next step.
- The classifier is trained on **synthetic** waveforms and validated on held-out
  synthetic data; live-sensor accuracy will be lower than the in-distribution
  numbers and is the next thing to characterize with real recordings.
- Physical eddy-current damping force on a moving conductive target has not yet
  been directly measured; damping is inferred from the characterized field and
  the standard eddy-current force approximation.

These limitations are inherited from the proof-of-concept stage of the hardware
and are stated so the scope is clear.

---

## License
MIT — see [LICENSE](LICENSE).
