#!/usr/bin/env python3
"""
Summary results for full and frozen fine-tuning:
  1. Kendall tau between ID and OOD model rankings, per task, then averaged per shift type
  2. Tasks where the best ID model is not the best OOD model, and the OOD points lost by picking on ID
  3. Overall ID rank vs overall OOD rank per model (task-relative scores, shifts weighted equally)
  4. ID -> OOD drop: absolute points and % of ID, overall, per model group, per model range, per shift type

Uses the same loading / cleaning / failed-adaptation rule as plot_ood_vs_er.py (must sit in the same folder).
Usage: python paper_numbers.py --full results_full.csv --head results_head.csv --out tables/ [--exclude OLMo-Earth Galileo]
"""
import argparse, os, sys
import numpy as np, pandas as pd
from scipy.stats import kendalltau
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_ood_vs_er import load, GROUP


def shift_weighted(k, col):
    """task-relative score per model: deviation from task mean, averaged within shift, shifts equal"""
    dev = k[col] - k.groupby("task")[col].transform("mean")
    return dev.groupby([k.model, k["shift"]]).mean().groupby(level=0).mean()


def analyse(d, tag, out):
    k = d[~d.failed].copy()
    print(f"\n######## {tag}: {k.task.nunique()} tasks, {k.model.nunique()} models, {len(k)} runs")

    # 1. Kendall tau per task / per shift
    rows = []
    for (s, t), g in k.groupby(["shift", "task"]):
        tau = kendalltau(g.ID, g.OOD).statistic
        bi, bo = g.loc[g.ID.idxmax()], g.loc[g.OOD.idxmax()]
        rows.append(dict(shift=s, task=t, n=len(g), tau=tau, best_ID=bi.model, best_OOD=bo.model,
                         regret=bo.OOD - bi.OOD, best_ID_OOD_rank=int(g.OOD.rank(ascending=False)[bi.name])))
    T = pd.DataFrame(rows)
    S = T.groupby("shift").tau.agg(["mean", "min", "max", "size"])
    print("\n1. Kendall tau(ID, OOD), mean of per-task values"); print(S.round(2).to_string())
    print(T[["shift", "task", "n", "tau"]].round(2).to_string(index=False))

    # 2. best ID != best OOD
    miss = T[T.best_ID != T.best_OOD]
    print(f"\n2. Best ID model is not best OOD model on {len(miss)} of {len(T)} tasks "
          f"({(miss.regret >= 1).sum()} with a gap of >= 1 OOD pt)")
    print(miss[["task", "best_ID", "best_OOD", "best_ID_OOD_rank", "regret"]].round(1).to_string(index=False))

    # 3. overall ID vs OOD rank
    R = pd.DataFrame({"ID": shift_weighted(k, "ID"), "OOD": shift_weighted(k, "OOD")})
    R["ID rank"] = R.ID.rank(ascending=False).astype(int); R["OOD rank"] = R.OOD.rank(ascending=False).astype(int)
    R["change"] = R["ID rank"] - R["OOD rank"]; R = R.sort_values("ID rank")
    print(f"\n3. Overall ID vs OOD rank (tau = {kendalltau(R.ID, R.OOD).statistic:.2f})"); print(R.round(2).to_string())

    # 4. drop
    k["drop"] = k.ID - k.OOD; k["rel"] = 100 * k["drop"] / k.ID; k["group"] = k.model.map(GROUP)
    pm = k.groupby(["model", "shift"])[["drop", "rel"]].mean().groupby("model").mean(); pm["group"] = pm.index.map(GROUP)
    sh = k.groupby(["shift", "task"])[["drop", "rel"]].mean().groupby("shift").mean()
    print(f"\n4. ID->OOD drop (per model: tasks averaged within shift, shifts equal)")
    print(f"   all models: mean {pm['drop'].mean():.1f} pts ({pm.rel.mean():.1f}% of ID); "
          f"model range {pm['drop'].min():.1f}-{pm['drop'].max():.1f} pts ({pm.rel.min():.1f}-{pm.rel.max():.1f}%)")
    print(pm.groupby("group")[["drop", "rel"]].agg(["mean", "min", "max"]).round(1).to_string())
    print(pm.sort_values("rel").round(1).to_string()); print("per shift type (mean over tasks):"); print(sh.round(1).to_string())

    for name, df in [("tau_per_task", T), ("tau_per_shift", S), ("id_vs_ood_rank", R), ("drop_per_model", pm), ("drop_per_shift", sh)]:
        df.round(3).to_csv(os.path.join(out, f"{name}_{tag}.csv"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", required=True); ap.add_argument("--head", required=True)
    ap.add_argument("--out", default="tables"); ap.add_argument("--floor", type=float, default=0.5)
    ap.add_argument("--exclude", nargs="*", default=["OLMo-Earth"])
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    for tag, path in [("full", a.full), ("head", a.head)]:
        analyse(load(path, a.floor, a.exclude, tag), tag, a.out)


if __name__ == "__main__":
    main()