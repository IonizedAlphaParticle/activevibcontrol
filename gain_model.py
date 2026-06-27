"""
Pathway 1: Adaptive PD Gain Model
---------------------------------
Learns to pick Kp/Kd as a function of operating point so the PD controller
behaves consistently across the NONLINEAR PWM->field->force curve the paper
characterized (B = 0.0644*exp(0.00706*OCR)+7.6024, force ~ B^2).

Why this is a real ML contribution and not bolted-on:
The achievable control authority (how much damping force a given gain command
actually produces) scales with B^2. So the SAME Kp/Kd produce very different
closed-loop behavior depending on duty cycle: gentle near the dead zone, violent
near full field. A single fixed gain pair therefore either overshoots at high
field or responds sluggishly at low field. The model adapts gains to the
operating point to keep settling fast and overshoot low everywhere.

All in simulation -> no hardware required.
"""

import numpy as np
from scipy.integrate import solve_ivp
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

# --- Paper constants (Section II.VI) ---
M = 0.5
K = 50.0
C_BASE = 0.1


def pwm_to_field(ocr):
    """Paper's fitted PWM(OCR) -> field (Gauss), R^2 = 0.975."""
    return 0.0644 * np.exp(0.00706 * ocr) + 7.6024


B_MIN = pwm_to_field(450.0)
B_MAX = pwm_to_field(950.0)


def control_authority(B):
    """Unit PD command -> actual force, scales with B^2. Normalized ~[0.2,1.0]."""
    return 0.2 + 0.8 * (B ** 2 - B_MIN ** 2) / (B_MAX ** 2 - B_MIN ** 2)


def simulate(kp, kd, gain_authority, x0=5.0, t_end=3.0, dt=0.01):
    def deriv(t, s):
        x, v = s
        e = -x
        u = kp * e - kd * v
        f_ctrl = gain_authority * u
        a = (f_ctrl - K * x - C_BASE * v) / M
        return [v, a]
    t_eval = np.arange(0, t_end, dt)
    sol = solve_ivp(deriv, (0, t_end), [x0, 0.0], t_eval=t_eval, method="RK23")
    x = sol.y[0]
    thresh = 0.05 * abs(x0)
    outside = np.where(np.abs(x) > thresh)[0]
    settle = sol.t[outside[-1]] if len(outside) else 0.0
    overshoot = max(0.0, -np.min(x) / abs(x0))
    return settle, overshoot


def cost(settle, overshoot):
    return settle + 3.0 * overshoot


def best_gains_for(authority):
    best = None
    for kp in np.linspace(2, 60, 12):
        for kd in np.linspace(0.5, 20, 12):
            s, o = simulate(kp, kd, authority)
            c = cost(s, o)
            if best is None or c < best[0]:
                best = (c, kp, kd, s, o)
    return best[1], best[2], best[3], best[4]


def build_dataset(n_points=30):
    X, y = [], []
    for ocr in np.linspace(450, 950, n_points):
        B = pwm_to_field(ocr)
        auth = control_authority(B)
        kp, kd, s, o = best_gains_for(auth)
        X.append([ocr, B, auth])
        y.append([kp, kd])
    return np.array(X), np.array(y)


if __name__ == "__main__":
    print("Building gain-tuning dataset from the paper's sim...")
    X, y = build_dataset()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
    model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=0)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    print("\n=== Gain model performance (held-out) ===")
    print(f"Kp MAE: {mean_absolute_error(yte[:,0], pred[:,0]):.2f}")
    print(f"Kd MAE: {mean_absolute_error(yte[:,1], pred[:,1]):.2f}")
    print("\nSample predictions:")
    for ocr in [500, 650, 800, 950]:
        B = pwm_to_field(ocr); auth = control_authority(B)
        kp, kd = model.predict([[ocr, B, auth]])[0]
        print(f"  OCR {ocr:4d}  B={B:5.1f}G  ->  Kp={kp:5.1f}  Kd={kd:5.1f}")
    print("\n=== Fixed-gain vs ML-adapted gain ===")
    fixed_auth = control_authority(pwm_to_field(950))
    fixed_kp, fixed_kd, _, _ = best_gains_for(fixed_auth)
    print(f"(fixed gains tuned at OCR 950: Kp={fixed_kp:.1f}, Kd={fixed_kd:.1f})\n")
    for ocr in [500, 650, 800, 950]:
        auth = control_authority(pwm_to_field(ocr))
        sf, of = simulate(fixed_kp, fixed_kd, auth)
        kp, kd = model.predict([[ocr, pwm_to_field(ocr), auth]])[0]
        sa, oa = simulate(kp, kd, auth)
        print(f"  OCR {ocr}:  FIXED settle={sf:.2f}s ov={of:.2f}  |  "
              f"ADAPTED settle={sa:.2f}s ov={oa:.2f}")
    joblib.dump(model, "../models/gain_model.joblib")
    print("\nSaved model -> gain_model.joblib")
