"""Makes the demo figure comparing how the system settles with fixed gains vs our adapted ones."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import joblib
from gain_model import best_gains_for, control_authority, pwm_to_field, M, K, C_BASE
from scipy.integrate import solve_ivp

model = joblib.load("../models/gain_model.joblib")

# --- styling ---
RED = "#C44E52"
BLUE = "#2B6CB0"
INK = "#2D2D2D"
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#888888",
    "axes.linewidth": 0.9,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.color": INK,
    "ytick.color": INK,
    "text.color": INK,
    "axes.labelcolor": INK,
})


def trajectory(kp, kd, authority, x0=5.0, t_end=3.0, dt=0.004):
    def deriv(t, s):
        x, v = s
        u = kp * (-x) - kd * v
        a = (authority * u - K * x - C_BASE * v) / M
        return [v, a]
    t_eval = np.arange(0, t_end, dt)
    sol = solve_ivp(deriv, (0, t_end), [x0, 0.0], t_eval=t_eval, method="RK23")
    return sol.t, sol.y[0]


def overshoot(x, x0=5.0):
    return max(0.0, -np.min(x) / abs(x0)) * 100


fixed_auth = control_authority(pwm_to_field(950))
fkp, fkd, _, _ = best_gains_for(fixed_auth)

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
fig.patch.set_facecolor("white")

for ocr, ax in zip([500, 800], axes):
    auth = control_authority(pwm_to_field(ocr))
    t1, x1 = trajectory(fkp, fkd, auth)
    kp, kd = model.predict([[ocr, pwm_to_field(ocr), auth]])[0]
    t2, x2 = trajectory(kp, kd, auth)

    ax.axhline(0, color="#bbbbbb", lw=1.0, zorder=1)
    ax.plot(t1, x1, color=RED, lw=2.4, zorder=3,
            label=f"Fixed gains  (Kp={fkp:.0f}, Kd={fkd:.0f})")
    ax.plot(t2, x2, color=BLUE, lw=2.4, zorder=4,
            label=f"ML-adapted  (Kp={kp:.0f}, Kd={kd:.0f})")

    # annotate the worst overshoot of the fixed curve
    imin = np.argmin(x1)
    ax.annotate(f"{overshoot(x1):.0f}% overshoot",
                xy=(t1[imin], x1[imin]),
                xytext=(t1[imin] + 0.45, x1[imin] - 0.15),
                fontsize=9, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.1))

    ax.set_title(f"OCR {ocr}   (B = {pwm_to_field(ocr):.0f} G)", pad=10, weight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Displacement (mm)")
    ax.grid(True, color="#e8e8e8", lw=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    leg = ax.legend(frameon=True, fontsize=9.5, loc="upper right",
                    handlelength=1.6, borderpad=0.7)
    leg.get_frame().set_edgecolor("#dddddd")
    leg.get_frame().set_linewidth(0.8)

fig.suptitle("Fixed vs ML-Adapted PD Gains  ·  Vibration Settling",
             fontsize=15, weight="bold", y=1.02)
fig.text(0.5, -0.04,
         "Same disturbance, same hardware. Fixed gains tuned for full field "
         "overshoot when the field is weaker; the adapted gains stay smooth.",
         ha="center", fontsize=9.5, color="#666666", style="italic")
fig.tight_layout()
fig.savefig("../figures/fig_adaptive_vs_fixed.png", dpi=140, bbox_inches="tight",
            facecolor="white")
print("saved ../figures/fig_adaptive_vs_fixed.png")
