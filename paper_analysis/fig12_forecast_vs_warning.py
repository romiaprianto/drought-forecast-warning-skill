"""Figure 12: warning skill (CSI, OR, D1) against continuous skill (SS vs persistence, GWETROOT), single cell.
Broken x-axis so that strongly negative skill scores do not compress the rest; no title or footnote in the graphic
(EMS: the caption carries them). Inputs: results/metrics_raw.csv, results/dss_validation.csv."""
import numpy as np, pandas as pd, matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
from _paths import RES, OUT
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.linewidth": 0.8})
COL = {"TCN": "#1f77b4", "LSTM": "#1a9e77", "Transformer": "#cc79a7", "RandomForest": "#e69f00", "Persistence": "#111111", "Climatology": "#7f7f7f"}
MK = {"TCN": "o", "LSTM": "^", "Transformer": "D", "RandomForest": "s", "Persistence": "X", "Climatology": "v"}
LAB = {"RandomForest": "Random forest"}
raw = pd.read_csv(RES / "metrics_raw.csv"); dv = pd.read_csv(RES / "dss_validation.csv")
cont = raw[raw.target == "GWETROOT"].groupby(["model", "horizon"]).SS_vs_persist.mean().reset_index()
m = cont.merge(dv[(dv.logic == "or") & (dv.level == "D1")][["model", "horizon", "CSI"]], on=["model", "horizon"]).dropna()
size = {1: 28, 7: 60, 14: 100, 30: 150}; cut = -2.4
far = m[m.SS_vs_persist < cut]; lo = far.SS_vs_persist.min()
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 5.6), sharey=True, gridspec_kw={"width_ratios": [1.4, 5], "wspace": 0.05})
for ax, sel in ((a1, m.SS_vs_persist < cut), (a2, m.SS_vs_persist >= cut)):
    pmax = m[m.model == "Persistence"].CSI.max(); pmin = m[m.model == "Persistence"].CSI.min()
    ax.axhspan(pmin, pmax, color="#555", alpha=0.08, zorder=0); ax.axhline(pmax, color="#444", lw=0.8, ls=":")
    for mod, g in m[sel].groupby("model"):
        ax.scatter(g.SS_vs_persist, g.CSI, s=[size[h] for h in g.horizon], c=COL[mod], marker=MK[mod],
                   edgecolors="white", linewidths=0.6, zorder=3, label=LAB.get(mod, mod))
a1.set_xlim(lo - 2.5, cut - 0.9); a2.set_xlim(cut, 0.9); a2.axvline(0, color="#888", lw=0.8, ls="--")
a1.spines["right"].set_visible(False); a2.spines["left"].set_visible(False); a2.tick_params(left=False)
a1.set_xticks([-40, -20, -5])
for ax, x0 in ((a1, 1), (a2, 0)):
    k = dict(transform=ax.transAxes, color="k", clip_on=False, lw=0.8)
    ax.plot((x0 - 0.04 * (4 if x0 else 1), x0 + 0.04 * (4 if x0 else 1)), (-0.012, 0.012), **k)
a2.text(0.88, pmax + 0.01, "persistence CSI", ha="right", va="bottom", fontsize=8, color="#444")
a1.set_ylabel("Warning skill: CSI (OR, \u2265 D1)"); a1.set_ylim(-0.03, 0.9)
fig.text(0.55, 0.02, "Continuous skill: skill score vs persistence (GWETROOT)", ha="center")
h, l = [], []
for mod in ["TCN", "LSTM", "Transformer", "RandomForest", "Persistence", "Climatology"]:
    h.append(plt.scatter([], [], s=60, c=COL[mod], marker=MK[mod])); l.append(LAB.get(mod, mod))
plt.sca(a2); a2.legend(h, l, title="marker size = lead time", fontsize=8, title_fontsize=8, loc="center left",
                                                   bbox_to_anchor=(0.02, 0.55), frameon=False)
fig.subplots_adjust(bottom=0.1)
fig.savefig(OUT / "fig12_forecast_vs_warning.png", dpi=300, bbox_inches="tight"); fig.savefig(OUT / "fig12_forecast_vs_warning.pdf", bbox_inches="tight")
print("fig10 ok | titik di panel kiri:", far[["model", "horizon", "SS_vs_persist"]].round(2).values.tolist())
