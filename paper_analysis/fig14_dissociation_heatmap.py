"""Figure 14: rank agreement (Spearman rho) between continuous skill and warning CSI by lead time and
drought level, for two targets and two spatial scales. Input: paper_outputs/dissociation_matrix.csv
(run dissociation_matrix.py first)."""
import numpy as np, pandas as pd, matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from _paths import OUT
M = pd.read_csv(OUT / "dissociation_matrix.csv"); M = M[M.horizon.astype(str) != "pooled"].assign(h=lambda x: x.horizon.astype(int))
leads, levels = [1, 7, 14, 30], ["D0", "D1", "D2", "D3"]
TL = {"GWETROOT": "Soil moisture (GWETROOT)", "SPEI_1": "Drought index (SPEI-1)"}
SL = {"multicell": "Multi-cell (pooled)", "regional": "Regional (\u226530% rule)"}
fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.6), constrained_layout=True)
norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
for i, tgt in enumerate(TL):
    for j, src in enumerate(SL):
        ax = axes[i, j]; g = M[(M.target == tgt) & (M.source == src)].pivot_table(index="level", columns="h", values="rho").reindex(index=levels, columns=leads)
        im = ax.imshow(g.values, cmap="RdBu", norm=norm, aspect="auto")
        for a_ in range(4):
            for b_ in range(4):
                v = g.values[a_, b_]
                if np.isfinite(v): ax.text(b_, a_, f"{v:+.2f}", ha="center", va="center", fontsize=9, color="white" if abs(v) > 0.55 else "#1a1a1a")
        ax.set_xticks(range(4)); ax.set_xticklabels(leads); ax.set_yticks(range(4)); ax.set_yticklabels(levels); ax.tick_params(length=0)
        for s in ax.spines.values(): s.set_visible(False)
        if i == 0: ax.set_title(SL[src], fontsize=10.5, weight="bold")
        if j == 0: ax.set_ylabel(f"{TL[tgt]}\n\nDrought level")
        if i == 1: ax.set_xlabel("Lead time (days)")
cb = fig.colorbar(im, ax=axes, shrink=0.85, ticks=[-1, -0.5, 0, 0.5, 1]); cb.set_label("Spearman \u03c1 (continuous skill vs warning CSI)")
fig.savefig(OUT / "fig14_dissociation_heatmap.png", dpi=300); fig.savefig(OUT / "fig14_dissociation_heatmap.pdf"); print("ok")
