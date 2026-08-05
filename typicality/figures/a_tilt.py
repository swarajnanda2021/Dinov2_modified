import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs(os.path.dirname(__file__), exist_ok=True)

# --- machinery (work in units of p_ref = 1) ---
# weight      w(p)   = (p + c)^(-a),      c = c_frac * p_ref = 0.25
# acceptance  A(p)   = w / w_max = (1 + p/c)^(-a) = (1 + 4p)^(-a)
# contrast    C(a)   = R^a   (R ~ 7.67 = measured effective density range)
c_frac = 0.25
c = c_frac                      # p_ref = 1
p = np.logspace(-2, 2, 600)     # density in units of p_ref
avals = [0.0, 0.5, 1.0, 2.0]
cols  = ["#9aa0a6", "#2a6fb0", "#a03030", "#6d28a8"]

plt.rcParams.update({"font.size": 11, "axes.edgecolor": "#666",
                     "axes.linewidth": 0.8, "figure.facecolor": "white"})
fig, ax = plt.subplots(1, 3, figsize=(16, 5.2))

# ================= Panel 1: weight, log-log =================
a1 = ax[0]
a1.axvspan(p.min(), c, color="#2a6fb0", alpha=0.05)
for a, col in zip(avals, cols):
    w = (p + c) ** (-a)
    a1.loglog(p, w, color=col, lw=2.4, label=f"a = {a:g}")
a1.axvline(c, color="#444", ls=":", lw=1.2)
a1.text(c*0.9, 0.02, "knee at  p̂ = c = c_frac·p_ref", rotation=90,
        va="bottom", ha="right", fontsize=9, color="#444")
a1.text(0.02, 11, "RARE\nweight saturates at\nceiling  w_max = c^(−a)",
        fontsize=9, color="#2a6fb0", va="top")
a1.text(6, 0.06, "COMMON\npower law,\nslope = −a", fontsize=9,
        color="#a03030", va="top", ha="left")
a1.annotate("", xy=(60, 0.11), xytext=(6, 0.5),
            arrowprops=dict(arrowstyle="->", color="#a03030", lw=1.3))
a1.set_xlabel("local density  p̂  (units of p_ref)")
a1.set_ylabel("weight  w = (p̂ + c)^(−a)")
a1.set_title("(1)  weight:  flat ceiling (rare)  →  slope −a (common)")
a1.legend(frameon=False, loc="upper right")
a1.grid(alpha=0.15, which="both")

# ================= Panel 2: acceptance, semilog-x =================
a2 = ax[1]
for a, col in zip(avals, cols):
    A = (1 + p / c) ** (-a)
    a2.semilogx(p, A, color=col, lw=2.4, label=f"a = {a:g}")
    if a > 0:
        a2.plot(1.0, (1 + 1/c) ** (-a), "o", color=col, ms=7, zorder=5)
a2.axvline(1.0, color="#444", ls=":", lw=1.2)
a2.text(1.05, 0.96, "p̂ = p_ref\n(median tile)", fontsize=9, color="#444", va="top")
a2.text(0.012, 1.02, "a = 0 : accept everything (no thinning)", fontsize=9, color="#9aa0a6")
a2.text(3, 0.62, "higher a →\ncommon tiles\nrejected harder", fontsize=9, color="#a03030")
a2.set_ylim(-0.03, 1.08)
a2.set_xlabel("local density  p̂  (units of p_ref)")
a2.set_ylabel("acceptance  A = (1 + p̂/c)^(−a)")
a2.set_title("(2)  acceptance:  keep-probability vs density")
a2.legend(frameon=False, loc="center left")
a2.grid(alpha=0.15, which="both")

# ================= Panel 3: contrast vs a, semilog-y =================
a3 = ax[2]
aa = np.linspace(0, 2, 200)
R = 7.67
a3.semilogy(aa, R ** aa, color="#1a3f7a", lw=2.6)
for am, Cm, lab in [(0.5, 2.77, "a=0.5\n2.77×  (ESS 0.89)"),
                    (1.0, 7.68, "a=1.0\n7.68×  (ESS 0.67)")]:
    a3.plot(am, Cm, "o", color="#a03030", ms=9, zorder=5)
    a3.annotate(lab, xy=(am, Cm), xytext=(am-0.42, Cm*1.6), fontsize=9,
                color="#a03030",
                arrowprops=dict(arrowstyle="->", color="#a03030", lw=1.1))
a3.plot(2.0, R**2, "s", color="#6d28a8", ms=8)
a3.text(1.93, R**2*0.62, f"a=2\n{R**2:.0f}×", fontsize=9, color="#6d28a8", ha="right")
a3.set_ylim(0.9, R**2*1.6)
a3.text(0.05, 1.15, "contrast = R^a\n(rarest ÷ commonest gradient mass)\n"
        "doubling a SQUARES the contrast:\n7.68 ≈ 2.77²", fontsize=9.5, color="#1a3f7a", va="bottom")
a3.set_xlabel("tilt exponent  a")
a3.set_ylabel("weight contrast  (log)")
a3.set_title("(3)  contrast = R^a   (cost: ESS falls)")
a3.grid(alpha=0.15, which="both")

for a in ax:
    for s in ["top", "right"]:
        a.spines[s].set_visible(False)
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), "a_regimes.png")
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out)
