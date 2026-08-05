import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Thinning-mode behaviour, built on one toy density and the same weight law.
#   density  x = p_hat / p_ref ~ lognormal(median 1, sigma), c = c_frac = 0.25
#   weight       w(x) = (x + c)^(-a)
#   acceptance   A(x) = w / w_max = (1 + x/c)^(-a),   w_max = c^(-a)
#   thinned committed dist  = A*mu / E[A]   ==   weighted effective dist = w*mu / E[w]
# sigma = 1.0 reproduces the measured ESS almost exactly (0.89 / 0.68).
# ---------------------------------------------------------------------------
SIG, c = 1.0, 0.25
u = np.linspace(-8, 8, 6000)                 # u = ln x
x = np.exp(u)
mu = np.exp(-(u**2) / (2 * SIG**2)); mu /= mu.sum()     # p(u) du, the stream

def E(g):  return float((g * mu).sum())
def ess_w(a):
    w = (x + c) ** (-a); return E(w) ** 2 / E(w * w)
def acc(a):  return E((1 + x / c) ** (-a))
def chi(a):  return 1.0 / acc(a)

blue, red, grey, pur, dk = "#2a6fb0", "#a03030", "#9aa0a6", "#6d28a8", "#1a3f7a"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": "#666",
                     "axes.linewidth": 0.8, "figure.facecolor": "white"})
fig, ax = plt.subplots(2, 2, figsize=(15, 10.5))

# ===== (1) matched-mass reshaping: same committed diet, two mechanisms =====
p = ax[0, 0]
p.fill_between(x, mu / mu.max(), color=grey, alpha=0.25, label="raw stream  μ(p̂)")
for a, col in [(0.5, blue), (1.0, red)]:
    A = (1 + x / c) ** (-a)
    comm = A * mu; comm /= comm.max()
    p.plot(x, comm, color=col, lw=2.4, label=f"committed, a={a:g}")
# overlay: weighted-effective (line) == thinned-committed (markers) for a=1.0
a = 1.0
weff = (x + c) ** (-a) * mu; weff /= weff.max()   # w·μ ∝ A·μ, so this lands on the a=1 curve
sel = np.linspace(500, 5500, 22).astype(int)
p.plot(x[sel], weff[sel], "o", color="#111", ms=4, zorder=6,
       label="weighted effective ≡ thinned")
p.set_xscale("log"); p.set_xlim(0.02, 50)
p.axvline(1.0, color="#444", ls=":", lw=1); p.text(1.06, 0.9, "p_ref", fontsize=9, color="#444")
p.annotate("tilt moves mass\ntoward the rare tail", xy=(0.12, 0.62), xytext=(0.03, 0.85),
           fontsize=9.5, color=red, arrowprops=dict(arrowstyle="->", color=red, lw=1.2))
p.set_xlabel("local density  p̂  (units of p_ref)"); p.set_ylabel("share of committed tiles (scaled)")
p.set_title("(1)  matched mass: thinning and weighting reshape the SAME diet")
p.legend(frameon=False, fontsize=9, loc="upper right"); p.grid(alpha=0.15, which="both")

# ===== (2) ESS vs a: the weighted road's cost; thinning stays at 1 =====
p = ax[0, 1]
aa = np.linspace(0, 2, 160); ew = np.array([ess_w(a) for a in aa])
p.fill_between(aa, ew, 1.0, color=blue, alpha=0.10)
p.plot(aa, ew, color=blue, lw=2.6, label="weighted arm  ESS = (Σw)²/(B·Σw²)")
p.axhline(1.0, color=red, lw=2.6, label="thinned arm  ESS = 1  (unit weights)")
for am, em in [(0.5, 0.89), (1.0, 0.67)]:
    p.plot(am, em, "o", color=dk, ms=8, zorder=5)
    p.annotate(f"measured {em:.2f}", xy=(am, em), xytext=(am + 0.05, em - 0.12),
               fontsize=9, color=dk, arrowprops=dict(arrowstyle="->", color=dk, lw=1))
p.text(1.15, 0.80, "this gap is the\neffective batch the\nweighted arm loses",
       fontsize=9.5, color=blue)
p.set_ylim(0.45, 1.03); p.set_xlabel("tilt exponent  a"); p.set_ylabel("effective sample fraction")
p.set_title("(2)  ESS: the weighted cost  (thinning pays nothing here)")
p.legend(frameon=False, fontsize=9, loc="lower left"); p.grid(alpha=0.15)

# ===== (3) over-draw chi vs a: the thinning cost; weighting needs none =====
p = ax[1, 0]
ch = np.array([chi(a) for a in aa])
p.plot(aa, ch, color=red, lw=2.6, label="thinned arm  χ = 1/E[A]  (pool per commit)")
p.axhline(1.0, color=blue, lw=2.6, label="weighted arm  χ = 1  (no over-draw)")
for am, cm in [(0.5, 2.0), (1.0, 3.3)]:
    p.plot(am, cm, "o", color=dk, ms=8, zorder=5)
    p.annotate(f"measured χ≈{cm:g}", xy=(am, cm), xytext=(am - 0.46, cm + 0.35),
               fontsize=9, color=dk, arrowprops=dict(arrowstyle="->", color=dk, lw=1))
for ov, lab in [(3, "oversample 3×"), (6, "oversample 6×")]:
    p.axhline(ov, color="#bbb", ls="--", lw=1); p.text(0.02, ov + 0.06, lab, fontsize=8.5, color="#888")
p.text(1.02, 1.35, "need pool ≥ χ·N\nto fill the batch →\na=1 wants 6×", fontsize=9.5, color=red)
p.set_ylim(0.8, 6.4); p.set_xlabel("tilt exponent  a"); p.set_ylabel("over-draw factor  χ")
p.set_title("(3)  over-draw: the thinning cost  (weighting pays nothing here)")
p.legend(frameon=False, fontsize=9, loc="upper left"); p.grid(alpha=0.15)

# ===== (4) fill statistics: why a=1.0 needs the bigger pool =====
p = ax[1, 1]
N = 256
def survN(pool, accr):
    m = pool * accr; s = np.sqrt(pool * accr * (1 - accr))
    xs = np.linspace(m - 4 * s, m + 4 * s, 400)
    return xs, np.exp(-(xs - m) ** 2 / (2 * s * s))
for (accr, ov, col, lab) in [(0.49, 3, blue, "a=0.5, 3× pool (768)"),
                             (0.30, 3, red, "a=1.0, 3× pool (768)"),
                             (0.30, 6, pur, "a=1.0, 6× pool (1536)")]:
    xs, ys = survN(256 * ov, accr); p.plot(xs, ys / ys.max(), color=col, lw=2.4, label=lab)
p.axvline(N, color="#111", lw=1.6); p.text(N + 6, 0.5, "N = 256\n(target)", fontsize=9)
p.axvspan(0, N, color="#a03030", alpha=0.05)
p.text(120, 0.9, "under-fill\n(top-up fires,\nmatched mass breaks)", fontsize=9, color=red, ha="center")
p.set_xlim(120, 520); p.set_ylim(0, 1.08)
p.set_xlabel("survivors admitted per step"); p.set_ylabel("relative frequency")
p.set_title("(4)  fill statistics: a=1.0 at 3× lands below N, needs 6×")
p.legend(frameon=False, fontsize=9, loc="upper right"); p.grid(alpha=0.15)

for row in ax:
    for a_ in row:
        for s_ in ["top", "right"]:
            a_.spines[s_].set_visible(False)
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), "thinning_behavior.png")
fig.savefig(out, dpi=125, bbox_inches="tight"); print("saved", out)
