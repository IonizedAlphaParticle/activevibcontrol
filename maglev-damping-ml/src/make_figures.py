"""This makes the demo figure comparing how the system settles with fixed gains vs our adapted ones."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from gain_model import (simulate, best_gains_for, control_authority,
                        pwm_to_field, M, K, C_BASE)
from scipy.integrate import solve_ivp

model = joblib.load("../models/gain_model.joblib")

def trajectory(kp, kd, authority, x0=5.0, t_end=3.0, dt=0.005):
    def deriv(t, s):
        x, v = s
        u = kp*(-x) - kd*v
        a = (authority*u - K*x - C_BASE*v)/M
        return [v, a]
    t_eval = np.arange(0, t_end, dt)
    sol = solve_ivp(deriv, (0, t_end), [x0, 0.0], t_eval=t_eval, method="RK23")
    return sol.t, sol.y[0]

# fixed gains tuned at full field
fixed_auth = control_authority(pwm_to_field(950))
fkp, fkd, _, _ = best_gains_for(fixed_auth)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ocr, ax in zip([500, 800], axes):
    auth = control_authority(pwm_to_field(ocr))
    t1, x1 = trajectory(fkp, fkd, auth)
    kp, kd = model.predict([[ocr, pwm_to_field(ocr), auth]])[0]
    t2, x2 = trajectory(kp, kd, auth)
    ax.axhline(0, color="#ccc", lw=0.8)
    ax.plot(t1, x1, label=f"Fixed gains (Kp={fkp:.0f}, Kd={fkd:.0f})", color="#d1495b", lw=2)
    ax.plot(t2, x2, label=f"ML-adapted (Kp={kp:.0f}, Kd={kd:.0f})", color="#2e86ab", lw=2)
    ax.set_title(f"OCR {ocr}  (B = {pwm_to_field(ocr):.0f} G)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Displacement (mm)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
fig.suptitle("Fixed vs ML-Adapted PD Gains: Vibration Settling", fontweight="bold")
fig.tight_layout()
fig.savefig("../figures/fig_adaptive_vs_fixed.png", dpi=130)
print("saved ../figures/fig_adaptive_vs_fixed.png")
