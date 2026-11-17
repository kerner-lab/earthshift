#!/usr/bin/env python3
"""
Overall OOD score vs overall effective robustness (ER) per model,
full fine-tuning and frozen backbone.

Method
  * Failed-adaptation runs (ID < FLOOR x task median ID) are excluded from everything.
  * Baseline: per shift type, a linear fit OOD = b*ID + a over all kept runs (all models, all tasks).
  * ER = OOD - baseline, then centred within each task (0 = the average model on that task).
  * OOD is taken relative to each task's mean and put back on the average-task scale,
    so models that are missing different tasks stay comparable.
  * Per model: tasks are averaged within each shift type, then shift types are weighted equally.

Usage
  python plot_ood_vs_er.py --full results_full.csv --head results_head.csv --out figures/
  optional: --floor 0.5  --exclude OLMo-Earth  --labels labels.json  --name fig_ood_vs_er

labels.json (optional) hand-places labels in data coordinates; any model not listed is placed
automatically:  {"Frozen backbone": {"Clay": [37.0, 2.75, "center"]}, "Full fine-tuning": {}}
"""
import argparse, json, math, os, re, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

NAME = {"clay": "Clay", "clip": "CLIP", "croma": "CROMA", "dinov3_large_sat": "DINOv3", "dofa": "DOFA",
        "prithvi": "Prithvi-2.0", "resnet50_imgnet": "R50-imgnet",
        "resnet_random": "R50-rnd", "terrafm": "TerraFM", "terramind": "TerraMind", "vit_imgnet": "ViT-imgnet",
        "vit_random": "ViT-rnd", "olmo-earth": "OLMo-Earth"}
GROUP = {"Clay": "GFM", "CROMA": "GFM", "DINOv3": "GFM", "DOFA": "GFM",
         "Prithvi-2.0": "GFM", "TerraFM": "GFM", "TerraMind": "GFM", "CLIP": "Generic", "R50-imgnet": "Generic",
         "ViT-imgnet": "Generic", "R50-rnd": "Random", "ViT-rnd": "Random", "OLMo-Earth": "GFM"}
# task -> shift type. Keys are normalised (lower case, letters and digits only) so spelling variants match.
SHIFT_RAW = {"RESISC45-UCMerced": "Scale", "UCMerced-RESISC45": "Scale",
             "Chesapeake-NAIP-Landsat": "Scale", "Chesapeake-Landsat-NAIP": "Scale",
             "BenV2-S2-S1": "Sensor", "Sen1Floods11-S2-S1": "Sensor", "RGB - RGE1": "Sensor",
             "Geo Germany-Cambodia": "Geographic", "Geo Germany-Denmark": "Geographic",
             "DeepGlobe-DFC2022": "Geographic", "BRIGHT": "Geographic",
             "SA Temporal": "Temporal", "Germany Temporal": "Temporal", "ftw-all-window": "Temporal",
             "France RPG 5year": "Temporal",
             "CloudSEN-L2A-L1C": "Processing", "BenV2-L2A-L1C": "Processing", "FTW-AI4Boundaries": "Processing"}
norm = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())
SHIFT = {norm(k): v for k, v in SHIFT_RAW.items()}
CANON = {norm(k): k for k in SHIFT_RAW}                      # one display name per task

INK, MUTED, GRID = "#1b2129", "#5b6570", "#d9dee3"
COL = {"GFM": "#2a78d6", "Generic": "#eb6834", "Random": "#1baf7a"}
FAM = {"GFM": "Geospatial FM", "Generic": "Generic pretrained", "Random": "Random init"}


def num(v):
    """'83.79%' -> 83.79 ; 0.8379 -> 83.79 ; blank/text -> NaN"""
    if pd.isna(v): return np.nan
    s = str(v).strip()
    if s == "": return np.nan
    try: return float(s[:-1]) if s.endswith("%") else float(s) * 100
    except ValueError: return np.nan


def load(path, floor, exclude, tag):
    raw = pd.read_csv(path, dtype=str, encoding="utf-8", encoding_errors="replace")
    raw = raw[raw.Model.notna() & raw.Task.notna()].copy()
    # repair rows shifted one column left (ID landed in 'Std Val Acc', OOD in 'Std Test ID Acc')
    # signature: a negative value in 'Std Test OOD Acc' (it holds Delta) or an empty 'Avg Test ID Acc' next to filled neighbours
    f = lambda c: pd.to_numeric(raw[c], errors="coerce")
    shifted = (f("Std Test OOD Acc") < 0) | (raw["Avg Test ID Acc"].isna() & raw["Std Val Acc"].notna() & raw["Std Test ID Acc"].notna())
    if shifted.any():
        print(f"[{tag}] WARNING: {shifted.sum()} rows look shifted one column left; realigning: "
              f"{sorted(raw.loc[shifted, 'Task'].unique())}", file=sys.stderr)
        cols = ["Std Val Acc", "Avg Test ID Acc", "Std Test ID Acc", "Avg Test OOD Acc", "Std Test OOD Acc"]
        vals = raw.loc[shifted, cols].to_numpy()
        # shifted layout: [ID, IDsd, OOD, OODsd, Delta] in [StdVal, AvgID, StdID, AvgOOD, StdOOD]
        raw.loc[shifted, "Avg Test ID Acc"] = vals[:, 0]; raw.loc[shifted, "Std Test ID Acc"] = vals[:, 1]
        raw.loc[shifted, "Avg Test OOD Acc"] = vals[:, 2]; raw.loc[shifted, "Std Test OOD Acc"] = vals[:, 3]
        raw.loc[shifted, "Std Val Acc"] = np.nan
    unknown_m = set(raw.Model.str.strip()) - set(NAME)
    if unknown_m: sys.exit(f"[{tag}] unknown model names: {unknown_m} (add them to NAME/GROUP)")
    unknown_t = {t for t in raw.Task if norm(t) not in SHIFT}
    if unknown_t: sys.exit(f"[{tag}] unknown tasks: {unknown_t} (add them to SHIFT_RAW)")
    d = pd.DataFrame({"task": raw.Task.map(lambda t: CANON[norm(t)]), "model": raw.Model.str.strip().map(NAME),
                      "seeds": pd.to_numeric(raw["Num Seeds"], errors="coerce"),
                      "ID": raw["Avg Test ID Acc"].map(num), "OOD": raw["Avg Test OOD Acc"].map(num)})
    empty = d.ID.isna() | d.OOD.isna() | (d.seeds.fillna(0) == 0)
    if empty.any():
        print(f"[{tag}] skipping {empty.sum()} rows with no results: "
              + ", ".join(f"{t}/{m}" for t, m in d.loc[empty, ["task", "model"]].itertuples(index=False)), file=sys.stderr)
    d = d[~empty & ~d.model.isin(exclude)].copy()
    dup = d.duplicated(["task", "model"], keep=False)
    if dup.any(): sys.exit(f"[{tag}] duplicate task/model rows: {d.loc[dup, ['task', 'model']].values.tolist()}")
    d["shift"] = d.task.map(lambda t: SHIFT[norm(t)])
    d["failed"] = d.ID < floor * d.groupby("task").ID.transform("median")
    for t, m, i in d.loc[d.failed, ["task", "model", "ID"]].itertuples(index=False):
        print(f"[{tag}] failed adaptation (excluded): {t} / {m} (ID {i:.1f})", file=sys.stderr)
    return d


def summarise(d):
    k = d[~d.failed].copy()
    fits = {s: np.polyfit(g.ID, g.OOD, 1) for s, g in k.groupby("shift")}
    k["ER"] = k.OOD - [fits[s][0] * x + fits[s][1] for s, x in zip(k["shift"], k.ID)]
    k["ER"] -= k.groupby("task").ER.transform("mean")
    k["dOOD"] = k.OOD - k.groupby("task").OOD.transform("mean")
    G = k.groupby("task").OOD.mean().groupby(k.groupby("task")["shift"].first()).mean().mean()
    per = k.groupby(["model", "shift"])[["dOOD", "ER"]].mean().unstack()
    M = pd.DataFrame({"OOD": per["dOOD"].mean(axis=1) + G, "ER": per["ER"].mean(axis=1),
                      "n_tasks": k.groupby("model").task.size(), "n_shifts": per["dOOD"].notna().sum(axis=1)})
    info = {"tasks": k.task.nunique(), "per_shift": k.groupby("shift").task.nunique().to_dict(),
            "fits": {s: (round(b, 3), round(a, 2)) for s, (b, a) in fits.items()}}
    return M.sort_values("OOD", ascending=False), info


def place_labels(fig, ax, pts):
    """Greedy placement: 16 directions at growing radii, first spot clear of placed labels and points."""
    fig.canvas.draw(); r = fig.canvas.get_renderer(); tr = ax.transData
    disp = {m: tr.transform(p) for m, p in pts.items()}; abox = ax.get_window_extent(r); placed = []
    def cost(bb):
        c = sum(max(0, min(bb.x1, p.x1) - max(bb.x0, p.x0)) * max(0, min(bb.y1, p.y1) - max(bb.y0, p.y0)) for p in placed)
        c += sum(400 for x, y in disp.values() if bb.x0 - 8 < x < bb.x1 + 8 and bb.y0 - 8 < y < bb.y1 + 8)
        if bb.x0 < abox.x0 or bb.x1 > abox.x1 or bb.y0 < abox.y0 or bb.y1 > abox.y1: c += 1e6
        return c
    crowd = {m: sum(1 for o in disp if o != m and math.dist(disp[m], disp[o]) < 40) for m in disp}
    for m in sorted(disp, key=lambda k: -crowd[k]):
        best = None
        for rad in (9, 16, 26, 38, 52, 68, 86):
            for k in range(16):
                ang = math.pi / 4 + k * math.pi / 8; dx, dy = math.cos(ang) * rad * 1.3, math.sin(ang) * rad
                ha = "left" if dx > 3 else "right" if dx < -3 else "center"; va = "bottom" if dy > 3 else "top" if dy < -3 else "center"
                t = ax.annotate(m, pts[m], xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=9, fontweight="bold")
                bb = t.get_window_extent(r).expanded(1.06, 1.15); c = cost(bb); t.remove()
                if best is None or c < best[0]: best = (c, dx, dy, ha, va, rad, bb)
                if c == 0: break
            if best[0] == 0: break
        _, dx, dy, ha, va, rad, bb = best
        ax.annotate(m, pts[m], xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=9, fontweight="bold",
                    color=INK, zorder=6, arrowprops=dict(arrowstyle="-", color="#7b858f", lw=0.8, shrinkA=1, shrinkB=6) if rad > 9 else None)
        placed.append(bb)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", required=True, help="CSV of full fine-tuning results")
    ap.add_argument("--head", required=True, help="CSV of frozen-backbone (head fine-tuning) results")
    ap.add_argument("--out", default="figures", help="output directory")
    ap.add_argument("--name", default="fig_ood_vs_er", help="output file stem")
    ap.add_argument("--floor", type=float, default=0.5, help="failed adaptation: ID < floor x task median ID")
    ap.add_argument("--exclude", nargs="*", default=["OLMo-Earth"], help="display names of models to leave out")
    ap.add_argument("--hide", nargs="*", default=[], help="display names of models kept in the baselines and "
                    "task centring but not plotted or listed")
    ap.add_argument("--labels", help="optional JSON of hand-placed labels")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    LBL = json.load(open(a.labels)) if a.labels else {}
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})

    R = {}
    for name, path, tag in [("Full fine-tuning", a.full, "full"), ("Frozen backbone", a.head, "head")]:
        M, info = summarise(load(path, a.floor, a.exclude, tag))
        M = M.drop(index=[m for m in a.hide if m in M.index]); R[name] = M
        print(f"\n{name}: {info['tasks']} tasks {info['per_shift']}\n  baselines (slope, intercept): {info['fits']}")
        print(M.round(2).to_string())
        M.round(3).to_csv(os.path.join(a.out, f"{a.name}_{tag}_values.csv"))

    ymin = min(M.ER.min() for M in R.values()); ymax = max(M.ER.max() for M in R.values())
    fig, axs = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    for ax, (name, M) in zip(axs, R.items()):
        ax.axhline(0, color=MUTED, lw=0.9, ls="--", zorder=1); ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
        nsh = M.n_shifts.max()
        lab = {m: (f"{m} ({M.loc[m, 'n_shifts']}/{nsh} shifts)" if M.loc[m, "n_shifts"] < nsh else m) for m in M.index}
        for m in M.index:
            c = COL[GROUP[m]]
            if M.loc[m, "n_shifts"] < nsh:   # missing whole shift types: hollow marker
                ax.plot(M.loc[m, "OOD"], M.loc[m, "ER"], "o", color="white", mec=c, mew=2, ms=9, zorder=5)
            else:
                ax.plot(M.loc[m, "OOD"], M.loc[m, "ER"], "o", color=c, mec="white", mew=1.5, ms=10, zorder=5)
        ax.set_xlim(M.OOD.min() - 2.5, M.OOD.max() + 2.5); ax.set_ylim(ymin - 1.2, ymax + 1.2)
        ax.set_title(name, loc="left", fontweight="bold", color=INK); ax.set_xlabel("Overall OOD score (pts)")
        auto = {}
        for m in M.index:
            px, py = M.loc[m, "OOD"], M.loc[m, "ER"]
            if m not in LBL.get(name, {}): auto[lab[m]] = (px, py); continue
            lx, ly, ha = LBL[name][m]; far = abs(lx - px) > 1.0 or abs(ly - py) > 0.45
            ax.annotate(lab[m], (px, py), xytext=(lx, ly), textcoords="data", ha=ha, va="center", fontsize=9, fontweight="bold",
                        color=INK, zorder=6, bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85),
                        arrowprops=dict(arrowstyle="-", color="#7b858f", lw=0.8, shrinkA=2, shrinkB=6) if far else None)
        if auto: place_labels(fig, ax, auto)
    axs[0].set_ylabel("Overall effective robustness (pts)")
    hand = [Line2D([], [], marker="o", ls="", color=COL[f], mec="white", markersize=9, label=FAM[f]) for f in COL]
    if any((M.n_shifts < M.n_shifts.max()).any() for M in R.values()):
        hand.append(Line2D([], [], marker="o", ls="", color="white", mec=MUTED, mew=2, markersize=8, label="Missing whole shift types"))
    fig.legend(handles=hand, loc="lower center", ncol=len(hand), frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for ext in ["png", "pdf"]:
        p = os.path.join(a.out, f"{a.name}.{ext}"); fig.savefig(p, dpi=200, bbox_inches="tight"); print("wrote", p)


if __name__ == "__main__":
    main()