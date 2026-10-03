"""Figure 7: skill score relative to persistence against lead time, single cell, for (a) root-zone soil moisture and
(b) SPEI-1. Mean +/- sd over five seeds; filled markers = significant improvement on persistence (Diebold-Mariano,
Benjamini-Hochberg-adjusted p < 0.05); values below -2 are drawn at the axis limit and labelled.
Inputs: results/metrics_raw.csv, results/diebold_mariano.csv."""
import numpy as np, pandas as pd, matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from _paths import RES, OUT
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 8.5, "axes.linewidth": 0.8})
MOD = [("TCN", "#0072B2", "o"), ("LSTM", "#009E73", "^"), ("Transformer", "#CC79A7", "D"), ("RandomForest", "#E69F00", "s")]
LAB = {"RandomForest": "Random forest"}; H = [1, 7, 14, 30]; YMIN, YMAX = -2.0, 0.75
raw = pd.read_csv(RES / "metrics_raw.csv"); dm = pd.read_csv(RES / "diebold_mariano.csv")
dm = dm[dm.comparison.str.endswith("_vs_Persistence")].assign(model=lambda x: x.comparison.str.replace("_vs_Persistence", ""))
fig, axs = plt.subplots(1, 2, figsize=(7.36, 3.44), sharey=True)
for ax, (tg, title, pl) in zip(axs, [("GWETROOT", "Root-zone soil moisture", "(a)"), ("SPEI_1", "SPEI-1", "(b)")]):
    ax.axhspan(YMIN, 0, color="#f2f2f2", zorder=0); ax.axhline(0, color="#444", lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.text(31.6, 0.015, "persistence", ha="right", va="bottom", fontsize=8, color="#444")
    offscale = {}
    for m, c, mk in MOD:
        g = raw[(raw.target == tg) & (raw.model == m)].groupby("horizon").SS_vs_persist
        mu, sd = g.mean().reindex(H), g.std().reindex(H)
        sig = {h: bool(dm[(dm.target == tg) & (dm.model == m) & (dm.horizon == h)].beats_persistence.iloc[0]) for h in H}
        ax.plot(H, np.maximum(mu.values, YMIN), color=c, lw=1.3, zorder=3, clip_on=True)
        for h in H:
            if mu[h] < YMIN:
                offscale.setdefault(h, []).append((m, c, mu[h])); continue
            ax.errorbar(h, mu[h], yerr=sd[h], color=c, lw=1.0, capsize=2.5, zorder=3)
            ax.plot(h, mu[h], marker=mk, ms=5.2, mfc=c if sig[h] else "white", mec=c, mew=1.3, zorder=4, clip_on=False)
    for h, items in offscale.items():
        for k, (m, c, v) in enumerate(items):
            ax.text(h - 0.7, YMIN + 0.05 + 0.12 * k, f"\u2193 {v:.0f}".replace("-", "\u2212"), color=c, fontsize=7.5, va="bottom", ha="left")
    ax.set_xticks(H); ax.set_xlim(0, 32); ax.set_ylim(YMIN, YMAX); ax.set_xlabel("Lead time (days)")
    ax.set_title(title, fontsize=10); ax.text(-0.14, 1.03, pl, transform=ax.transAxes, fontsize=10, weight="bold")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axs[0].set_ylabel("Skill score vs persistence"); axs[1].tick_params(labelleft=False)
axs[1].legend([Line2D([], [], color=c, marker=mk, mfc="white", mec=c, mew=1.3, lw=1.4, ms=5.2) for _, c, mk in MOD],
              [LAB.get(m, m) for m, _, _ in MOD], loc="lower right", frameon=False, fontsize=7.8)
fig.text(0.005, 0.01, "Filled markers: significantly better than persistence (Diebold-Mariano, FDR-controlled $p$ < 0.05). "
         "Arrows mark off-scale values.", fontsize=7.5, color="#666")
fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(OUT / "fig07_skill_vs_lead.png", dpi=600); fig.savefig(OUT / "fig07_skill_vs_lead.pdf")
print("ok")
