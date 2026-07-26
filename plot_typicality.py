#!/usr/bin/env python3
"""
plot_typicality.py — plot the typicality-bank metrics in the terminal.

Updated for the rev7 fixed-radius readout log format. Reads THREE generations of logs
and maps them onto the same canonical names, so old and new runs plot the same way:
  * REV7 (fixed-radius absolute readout): typ_p_median, typ_p_ref, typ_w_mean, typ_ess,
    typ_graduations, plus the bank-health keys (typ_s, typ_j, typ_h_over_L, typ_lam_*,
    typ_evict_over_q10, typ_underresolved).
  * REV5/6 (fixed-count percentile readout): typ_t_mean, typ_t_std, typ_w_mean, ...
  * legacy: typicality_s, typicality_admits, typicality_graduations, ...
The one-line `typ | ...` summary is ignored by the parser (it carries no `It <n>/`
prefix); this script reads the metric line above it.

KEY SIGNALS
  lam_spread     q90/q10 of the hit-rate = the density dynamic range. THE signal the
                 counters run on. If it heads below ~2.5 the bank can no longer grade
                 degrees of commonness.
  p_ref          the reference density scale (EMA of the batch-median p_hat). Sets the
                 weight floor c = c_frac * p_ref. Non-zero within a few steps of
                 activation; 0 => still cold-start (no modulation) or uninitialised.
  ess            effective fraction of the batch, sum(w)^2 / (sum(w^2) * B). THE health
                 signal for the modulation (replaces t_std, which sat at sqrt(1/12) by
                 construction and never moved). ~1 = near-uniform weights; heading below
                 ~0.5 => the loss is concentrated on a few tiles (raise the tilt only if
                 that is intended). Measured ~90% at a=0.5, ~69% at a=1.0.
  w_mean         mean absolute DINO weight w = 1/(p_hat + c)^a. Order 1-3, NOT below 1 --
                 absolute weights are not normalised to mean 1 (the loss divides by
                 sum(w), so only the spread of w acts).
  evict_z        (ln evict - ln q10) / (ln q90 - ln q10): the eviction alarm, NORMALISED
                 BY THE SPREAD. Negative = eviction confined below the bottom decile
                 (good). >= 0 = eviction reaching signatures with real counts.
                 WHY NOT evict/q10: as the rate distribution compresses, the minimum
                 necessarily approaches q10, so the raw ratio drifts toward 1 whether or
                 not eviction misbehaves. Use evict_z for the alarm, evict_over_q10 for
                 display.
  turn_ratio     turnover / n_eff, with n_eff = (1+eta)/(1-eta) the estimator's effective
                 sample size (~721 at half-life 250), fixed by the half-life alone and NOT
                 by stream length. Below ~2 the argmin eviction selects on noise.
  h_over_L       achieved pooling bandwidth in units of the correlation length; ~1 is the
                 target. Meaningless while `underresolved` is set. NOTE: since rev7 j and
                 h/L are DIAGNOSTICS only -- they no longer feed the readout, which sums
                 over the fixed radius R_rad = radius_mult * s. Read them to judge whether
                 the fixed radius is well-resolved, not as a readout knob.
  underresolved  1 => stored-signature spacing exceeds the fitted correlation length, so
                 no j reaches h = L. The fitted L (hence h/L) is then unreliable; the
                 remedy is a larger bank, not a j change.

NOTE on --avg: the running-average column for lam_q90 is polluted by an early newborn
edge case (a signature seeded S=0,E=0 and hit the same step gives lam = 1/eps). The
instantaneous window value is clean, so the default is correct.

Usage
-----
    python3 plot_typicality.py <run>                    # 4 "is it working" panels
    python3 plot_typicality.py <run> -g tune            # s, j, h/L, underresolved
    python3 plot_typicality.py <run> -g lam --overlay   # the hit-rate fan
    python3 plot_typicality.py <run> --status-only      # health check, no plots
    python3 plot_typicality.py <run> --watch 30         # live

Groups (-g/--group)
    key     turnover, grad_frac, replacements, hit_frac      (default)
            the four convergence signals: mean residence time (= M / graduations
            per step), the fraction of MISSING tiles that prove recurrent enough
            to be kept, cumulative bank replacements, and stream coverage
    health  evict_z, evict_over_q10, turn_ratio, ess, w_mean
    lam     lam_q10, lam_median, lam_q90                   (try --overlay)
    score   p_median, p_ref, w_mean, ess
    flow    hit_frac, grad_per_step, turnover, turn_ratio, reserve_residency
    bank    n_est, reserve, s
    tune    s, j, h_over_L, underresolved
    residence  turnover, median_life, surv_2k, replacements, grad_per_step
    all     everything available
"""

import argparse
import glob
import json
import math
import os
import re
import sys
import time

try:
    import plotext as plt
except ImportError:
    sys.exit("error: plotext not installed.\n"
             "  install it with:  python3 -m pip install --user plotext")


def find_logs(path):
    """All log files for a run, oldest-written first.

    Every SLURM restart opens a new *_0_log.out, so a single file is only
    "since the last restart". We return them all, ordered by mtime, and the
    loader supersedes duplicated iterations with the value from the newer
    file.
    """
    if os.path.isfile(path):
        return [path]
    if os.path.isdir(path):
        outs = glob.glob(os.path.join(path, "logs", "*_0_log.out"))
        outs = [f for f in outs if os.path.getsize(f) > 0]
        if outs:
            return sorted(outs, key=os.path.getmtime)
        for cand in ("logs/log.txt", "log.txt", "logs.txt", "logs/logs.txt"):
            full = os.path.join(path, cand)
            if os.path.isfile(full):
                return [full]
    sys.exit(f"error: could not find a log file at/under: {path}")


_IT_RE = re.compile(r"\bIt\s+(\d+)\s*/")
_NUM = r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
_METRIC_RE = re.compile(rf"([A-Za-z_][A-Za-z0-9_]*):\s*({_NUM})(?:\s*\(\s*({_NUM})\s*\))?")

ALIASES = {
    # rev7 fixed-radius readout format
    "typ_s": "s", "typ_j": "j", "typ_h_over_L": "h_over_L",
    "typ_n_est": "n_est", "typ_reserve": "reserve", "typ_hit_frac": "hit_frac",
    "typ_lam_q10": "lam_q10", "typ_lam_median": "lam_median",
    "typ_lam_q90": "lam_q90", "typ_lam_spread": "lam_spread",
    "typ_evict_over_q10": "evict_over_q10", "typ_underresolved": "underresolved",
    "typ_grad_per_step": "grad_per_step", "typ_turnover": "turnover",
    "typ_graduations": "graduations",
    "typ_p_median": "p_median", "typ_p_ref": "p_ref",
    "typ_w_mean": "w_mean", "typ_ess": "ess",
    # rev5/6 fixed-count percentile format (t_mean/t_std retired in rev7; kept for old runs)
    "typ_t_mean": "t_mean", "typ_t_std": "t_std",
    # legacy format
    "typicality_s": "s", "typicality_n_est": "n_est",
    "typicality_reserve": "reserve", "typicality_admits": "admits",
    "typicality_graduations": "graduations",
    "typicality_lam_q10": "lam_q10", "typicality_lam_median": "lam_median",
    "typicality_lam_q90": "lam_q90", "typicality_evict_lam": "evict_lam",
    "typicality_strong_core": "strong_core",
    "typicality_t_mean": "t_mean", "typicality_t_std": "t_std",
    "typicality_t_lt0p1": "t_lt0p1", "typicality_t_gt0p9": "t_gt0p9",
    "lr": "lr", "wd": "wd",
}


def parse_out_line(line, raw=True):
    m = _IT_RE.search(line)
    if not m:
        return None
    rec = {"iteration": float(m.group(1))}
    _, sep, body = line.partition(" : ")
    if not sep:
        body = line
    for name, window, avg in _METRIC_RE.findall(body):
        key = ALIASES.get(name)
        if key is None:
            continue
        val = window if (raw or not avg) else avg
        try:
            rec[key] = float(val)
        except ValueError:
            continue
    return rec if len(rec) > 1 else None


def parse_json_line(line):
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    out = {}
    for k, v in obj.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        kk = k[6:] if k.startswith("train_") else k
        if kk == "iteration":
            out["iteration"] = float(v)
        elif kk in ALIASES:
            out[ALIASES[kk]] = float(v)
    return out or None


def load_records(logfiles, raw=True):
    """Merge records across log files, newest write wins per iteration.

    Accepts a single path or a list of paths (oldest-written first). A run
    that was preempted and resumed replays iterations; the record from the
    later-written file supersedes the earlier one.
    """
    if isinstance(logfiles, str):
        logfiles = [logfiles]

    merged = {}
    for logfile in logfiles:
        with open(logfile, errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = parse_json_line(line) if line.startswith("{") else None
                if rec is None:
                    rec = parse_out_line(line, raw)
                if not rec:
                    continue
                it = rec.get("iteration")
                if it is None:
                    continue
                rec["_src"] = os.path.basename(logfile)
                merged[it] = rec          # later file supersedes earlier

    recs = [merged[k] for k in sorted(merged)]
    return recs


def augment(recs, batch, capacity, halflife):
    eta = 0.5 ** (1.0 / halflife)
    n_eff = (1.0 + eta) / (1.0 - eta)
    prev = None
    for r in recs:
        it = r.get("iteration")
        q10, q90 = r.get("lam_q10"), r.get("lam_q90")

        if "lam_spread" not in r and q10 and q90 and q10 > 0:
            r["lam_spread"] = q90 / q10
        if "evict_over_q10" not in r:
            ev = r.get("evict_lam")
            if ev is not None and q10 and q10 > 0:
                r["evict_over_q10"] = ev / q10

        adm = r.get("admits")
        if adm is not None and batch > 0 and "hit_frac" not in r:
            r["hit_frac"] = 1.0 - adm / batch
        if adm is None and r.get("hit_frac") is not None and batch > 0:
            adm = (1.0 - r["hit_frac"]) * batch
            r["admits"] = adm
        res = r.get("reserve")
        if res is not None and adm and adm > 0:
            r["reserve_residency"] = res / adm

        g = r.get("graduations")
        if "grad_per_step" not in r and it is not None and g is not None:
            if prev is not None:
                pit, pg = prev
                dit, dg = it - pit, g - pg
                if dit > 0 and dg >= 0:
                    r["grad_per_step"] = dg / dit
            prev = (it, g)
        gps = r.get("grad_per_step")
        if "turnover" not in r and gps and gps > 0:
            r["turnover"] = capacity / gps
        # of the tiles that MISS the bank, what fraction prove recurrent enough
        # to earn a permanent slot? Falls as the bank captures the recurrent
        # structure and the residual arrivals become one-offs.
        if gps is not None and adm and adm > 0:
            r["grad_frac"] = gps / adm

        turn = r.get("turnover")
        if turn:
            r["turn_ratio"] = turn / n_eff
            # Each graduation evicts one entry, so the per-entry hazard is
            # G/M per step and lifetimes are ~exponential with mean = turnover.
            r["median_life"] = math.log(2.0) * turn
            r["surv_2k"] = math.exp(-2000.0 / turn)
            r["surv_10k"] = math.exp(-10000.0 / turn)

        eo, sp = r.get("evict_over_q10"), r.get("lam_spread")
        if eo and sp and eo > 0 and sp > 1:
            r["evict_z"] = math.log(eo) / math.log(sp)

        r["n_eff"] = n_eff

    # cumulative bank replacements: integrate graduations/step over iterations
    cum, prev_it = 0.0, None
    for r in recs:
        it, gps = r.get("iteration"), r.get("grad_per_step")
        if it is not None and gps is not None:
            if prev_it is not None and it > prev_it:
                cum += gps * (it - prev_it) / capacity
            prev_it = it
            r["replacements"] = cum
    return recs


GROUPS = {
    "key":    ["turnover", "grad_frac", "replacements", "hit_frac"],
    "health": ["evict_z", "evict_over_q10", "turn_ratio", "ess", "w_mean"],
    "lam":    ["lam_q10", "lam_median", "lam_q90", "lam_spread"],
    "score":  ["p_median", "p_ref", "w_mean", "ess"],
    "flow":   ["hit_frac", "grad_per_step", "turnover", "turn_ratio",
               "reserve_residency"],
    "bank":   ["n_est", "reserve", "s"],
    "tune":   ["s", "j", "h_over_L", "underresolved"],
    "residence": ["turnover", "median_life", "surv_2k", "replacements",
                  "grad_per_step", "grad_frac"],
}
GROUPS["all"] = (GROUPS["bank"] + ["j", "h_over_L"] + GROUPS["lam"] +
                 ["lam_spread", "evict_over_q10", "evict_z"] + GROUPS["flow"] +
                 GROUPS["score"])


def keys_present(recs):
    keys, seen = [], set()
    for r in recs:
        for k, v in r.items():
            if k in seen or isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            seen.add(k); keys.append(k)
    return keys


def series(recs, key, xkey="iteration"):
    xs, ys = [], []
    for i, r in enumerate(recs):
        y = r.get(key)
        if isinstance(y, bool) or not isinstance(y, (int, float)):
            continue
        xs.append(float(r.get(xkey, i))); ys.append(float(y))
    return xs, ys


def ema(ys, f):
    if f <= 0:
        return ys
    out, m = [], None
    for y in ys:
        m = y if m is None else f * m + (1 - f) * y
        out.append(m)
    return out


def fmt(v):
    if v == 0:
        return "0"
    a = abs(v)
    if a >= 1e4 or a < 1e-3:
        return f"{v:.2e}"
    if a >= 100:
        return f"{v:.1f}"
    return f"{v:.3f}"


def nice_ticks(lo, hi, target=8):
    """Ticks at round numbers: multiples of 1, 2, 5 or 10 x a power of ten."""
    if not (hi > lo) or not all(map(math.isfinite, (lo, hi))):
        return [lo], 1.0
    raw = (hi - lo) / max(1, target)
    mag = 10.0 ** math.floor(math.log10(raw))
    step = 10 * mag
    for m in (1, 2, 5, 10):
        if raw <= m * mag:
            step = m * mag
            break
    ticks, v = [], math.ceil(lo / step) * step
    while v <= hi + 1e-9 * step:
        ticks.append(round(v, 10))
        v += step
    return (ticks or [lo, hi]), step


def tick_labels(ticks, step):
    """Decimals are set by the STEP so no label is silently rounded."""
    dec, s = 0, abs(step)
    while dec < 6 and abs(s - round(s)) > 1e-9:
        s *= 10
        dec += 1
    if dec == 0:
        return [f"{int(round(t))}" for t in ticks]
    return [f"{t:.{dec}f}" for t in ticks]


def apply_ticks(sp, xs, ys, ny=5, nx=7):
    # Vertical character rows are scarce; if more ticks are requested than fit,
    # plotext silently DROPS labels and the survivors then look unevenly spaced.
    # Ask for few enough that none are dropped.
    xt, xstep = nice_ticks(min(xs), max(xs), nx)
    yt, ystep = nice_ticks(min(ys), max(ys), ny)
    sp.xticks(xt, tick_labels(xt, xstep))
    sp.yticks(yt, tick_labels(yt, ystep))


def last_val(recs, key):
    for r in reversed(recs):
        v = r.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return v
    return None


def status(recs, args):
    it = last_val(recs, "iteration")
    n_eff = last_val(recs, "n_eff")
    print(f"\n=== typicality bank @ iter {int(it) if it else '?'} "
          f"(batch={args.batch}, M={args.capacity}, half-life={args.halflife}, "
          f"n_eff={fmt(n_eff) if n_eff else '?'}) ===")

    rows, warns = [], []

    def row(label, key, note=""):
        v = last_val(recs, key)
        if v is not None:
            rows.append((label, fmt(v), note))
        return v

    row("hit radius s", "s")
    row("pool j (diag)", "j", "resolution diagnostic, not a readout knob")
    row("h / L (diag)", "h_over_L", "target ~1")
    ur = row("underresolved", "underresolved", "1 = spacing > L; L unreliable")
    n_est = row("n_est", "n_est")
    row("reserve", "reserve")
    resid = row("reserve residency", "reserve_residency", "steps on probation")

    spread = row("lam spread q90/q10", "lam_spread", "the density signal")
    lam3 = [last_val(recs, k) for k in ("lam_q10", "lam_median", "lam_q90")]
    if all(v is not None for v in lam3):
        rows.append(("lam q10/med/q90", " / ".join(fmt(v) for v in lam3), ""))
    row("evict / q10", "evict_over_q10", "display only (drifts with compression)")
    ez = row("evict_z", "evict_z", "ALARM if >= 0 (spread-normalised)")

    hitf = row("hit fraction", "hit_frac")
    row("graduations / step", "grad_per_step")
    row("graduation fraction", "grad_frac", "of misses, share kept")
    row("turnover (steps)", "turnover", "mean entry lifetime")
    row("median lifetime", "median_life", "= ln2 x turnover")
    row("survive 2k steps", "surv_2k", "predicted, uniform hazard")
    row("survive 10k steps", "surv_10k", "predicted, uniform hazard")
    row("bank replacements", "replacements", "since activation")
    tr = row("turnover / n_eff", "turn_ratio", "ALARM if < 2")

    # ---- fixed-radius readout / weight (rev7) ----
    pmed = row("p_median density", "p_median", "batch-median p_hat")
    pref = row("p_ref (EMA scale)", "p_ref", "weight ref; c = c_frac * p_ref")
    ess = row("ESS fraction", "ess", "effective batch; < 0.5 = concentrated")
    row("mean DINO weight", "w_mean", "absolute; order 1-3, not < 1")
    # legacy rev5/6 percentile readout, shown only if present in the log
    row("t_mean (legacy)", "t_mean", "rev5/6 percentile readout only")
    row("t_std (legacy)", "t_std", "rev5/6 percentile readout only")

    w = max(len(r[0]) for r in rows) if rows else 0
    for label, val, note in rows:
        print(f"  {label:<{w}}  {val:>12}   {note}")

    if n_est is not None and n_est < args.capacity * 0.999:
        warns.append(f"bank UNDERFILLED: n_est={fmt(n_est)} < M={args.capacity}")
    if spread is not None and spread < 2.5:
        warns.append(f"lam spread {fmt(spread)} < 2.5: density discrimination is thin")
    if ez is not None and ez >= 0:
        warns.append(f"evict_z {fmt(ez)} >= 0: eviction reaching signatures with "
                     f"real counts")
    if tr is not None and tr < 2:
        warns.append(f"turnover/n_eff {fmt(tr)} < 2: slots evicted before their "
                     f"counters settle (argmin selects on noise)")
    if hitf is not None and hitf < 0.5:
        warns.append(f"hit fraction {fmt(hitf)} < 0.5: counters starved")
    if pref is not None and pref == 0 and pmed == 0:
        warns.append("p_ref = 0 and p_median = 0: still cold-start (bank full but counters "
                     "immature) -- NO modulation applied yet; this is expected right after "
                     "activation, not a fault")
    if ess is not None and ess < 0.5:
        warns.append(f"ESS {fmt(ess)} < 0.5: the DINO loss is concentrated on a few tiles "
                     f"(the tilt a may be too high for the density spread)")
    if resid is not None and resid < 10:
        warns.append(f"reserve residency {fmt(resid)} steps: reserve flushed by "
                     f"CAPACITY, not by T_need")
    if ur:
        warns.append("underresolved=1: signature spacing exceeds the fitted L, so no "
                     "j reaches h=L. Treat the fitted L and h/L as unreliable; the "
                     "remedy is a larger bank, not a j change.")

    print("\n  ! " + "\n  ! ".join(warns) if warns else
          "\n  all tracked thresholds nominal")
    print()


COLORS = ["cyan", "orange", "green", "magenta", "red", "blue", "yellow"]


def draw(recs, metrics, args):
    plt.clf(); plt.theme("clear")
    height = args.height if args.height > 0 else None

    if args.overlay:
        plt.subplots(1, 1)
        for i, key in enumerate(metrics):
            xs, ys = series(recs, key, args.xkey)
            if not xs:
                continue
            if args.last > 0:
                xs, ys = xs[-args.last:], ys[-args.last:]
            plt.plot(xs, ema(ys, args.smooth), label=key,
                     color=COLORS[i % len(COLORS)])
        if args.logy:
            plt.yscale("log")
        else:
            ax_x = [v for k in metrics for v in series(recs, k, args.xkey)[0]]
            ax_y = [v for k in metrics for v in series(recs, k, args.xkey)[1]]
            if ax_x and ax_y:
                apply_ticks(plt, ax_x, ax_y)
        plt.title(f"typicality vs {args.xkey}  ({len(recs)} records)")
        plt.xlabel(args.xkey)
        if height:
            plt.plotsize(None, height)
        plt.show(); return

    metrics = [m for m in metrics if series(recs, m, args.xkey)[0]]
    if not metrics:
        sys.exit("error: no matching metrics with data. Try --list.")
    ncol = max(1, args.cols)
    nrow = -(-len(metrics) // ncol)          # ceil
    plt.subplots(nrow, ncol)
    for i, key in enumerate(metrics):
        xs, ys = series(recs, key, args.xkey)
        if args.last > 0:
            xs, ys = xs[-args.last:], ys[-args.last:]
        raw_last = ys[-1]
        ys = ema(ys, args.smooth)
        sp = plt.subplot(i // ncol + 1, i % ncol + 1)
        sp.plot(xs, ys, color=COLORS[i % len(COLORS)])
        if args.logy:
            sp.yscale("log")
        if not args.logy:
            apply_ticks(sp, xs, ys)
        sp.title(f"{key}  last={fmt(raw_last)} min={fmt(min(ys))} max={fmt(max(ys))}")
        sp.xlabel(args.xkey)
    if height:
        plt.plotsize(None, height * nrow)
    plt.show()


def main():
    ap = argparse.ArgumentParser(
        description="Plot typicality-bank metrics in the terminal via plotext.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("path")
    ap.add_argument("-g", "--group", default="key",
                    help="key|health|lam|score|flow|bank|tune|residence|all")
    ap.add_argument("-m", "--metrics", default="", help="CSV substrings")
    ap.add_argument("-s", "--smooth", type=float, default=0.0)
    ap.add_argument("--avg", action="store_true")
    ap.add_argument("--overlay", action="store_true")
    ap.add_argument("--last", type=int, default=0)
    ap.add_argument("--logy", action="store_true")
    ap.add_argument("--height", type=int, default=0)
    ap.add_argument("-c", "--cols", type=int, default=2,
                    help="subplot columns (default 2; use 1 for a single column)")
    ap.add_argument("--watch", type=float, default=0.0)
    ap.add_argument("--xkey", default="iteration")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--capacity", type=int, default=8192)
    ap.add_argument("--halflife", type=int, default=250)
    ap.add_argument("--no-status", action="store_true")
    ap.add_argument("--status-only", action="store_true")
    ap.add_argument("--list", action="store_true", dest="do_list")
    args = ap.parse_args()

    logfiles = find_logs(args.path)
    if len(logfiles) > 1:
        print(f"merged {len(logfiles)} log files: "
              f"{', '.join(os.path.basename(f) for f in logfiles)}")

    def get_recs():
        recs = load_records(logfiles, raw=not args.avg)
        if not recs:
            sys.exit(f"error: no records parsed from {logfiles}")
        return augment(recs, args.batch, args.capacity, args.halflife)

    if args.do_list:
        recs = get_recs()
        print(f"logs: {len(logfiles)} file(s)   ({len(recs)} records, "
              f"iter {int(recs[0]['iteration'])}–{int(recs[-1]['iteration'])})")
        for k in keys_present(recs):
            print(f"  {k:<24} {len(series(recs, k, args.xkey)[0])} pts")
        return

    def render():
        recs = get_recs()
        if args.metrics.strip():
            subs = [x.strip() for x in args.metrics.split(",") if x.strip()]
            metrics = [k for k in keys_present(recs)
                       if any(x in k for x in subs) and k != args.xkey]
        else:
            if args.group not in GROUPS:
                sys.exit(f"error: unknown group. choose from: {', '.join(GROUPS)}")
            metrics = GROUPS[args.group]
        have = [m for m in metrics if series(recs, m, args.xkey)[0]]
        if not have:
            sys.exit("error: no typicality metrics with data (is the run past "
                     "T_warm?). Try --list.")
        if not args.status_only:
            draw(recs, have, args)
        if not args.no_status:
            status(recs, args)

    if args.watch > 0:
        try:
            while True:
                plt.clt(); render()
                print(f"[watching {len(logfiles)} log(s) — "
                      f"refresh {args.watch:g}s — Ctrl-C]")
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\nstopped.")
    else:
        render()


if __name__ == "__main__":
    main()
