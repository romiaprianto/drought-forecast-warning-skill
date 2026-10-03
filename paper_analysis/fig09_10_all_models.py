"""Figures 9 (soil moisture) and 10 (SPEI-1): forecast against observation at the 14-day lead for all four
learned models, single cell, same design as Figure 8. Forecasts are averaged over the five seeds. Input: results/predictions.csv."""
import sys, numpy as np, pandas as pd, matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from _paths import RES, OUT as _OUT
PRED = RES / "predictions.csv"; OUT = str(_OUT)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.linewidth": 0.8, "mathtext.default": "it"})
pr = pd.read_csv(PRED, parse_dates=["date"]); pr = pr[pr.horizon == 14]
MODELS = [("TCN", "#1f77b4"), ("LSTM", "#1a9e77"), ("Transformer", "#cc79a7"), ("RandomForest", "#e69f00")]
LAB = {"RandomForest": "Random Forest"}
for tg, name, tag in [("GWETROOT", "Root-zone soil moisture", "09"), ("SPEI_1", "SPEI-1", "10")]:
    fig = plt.figure(figsize=(7.4, 9.2))
    gs = fig.add_gridspec(4, 2, width_ratios=[2.4, 1], hspace=0.45, wspace=0.28, left=0.09, right=0.98, top=0.93, bottom=0.06)
    b = pr[pr.target == tg]; per = b[b.model == "Persistence"].sort_values("date")
    allv = pd.concat([b.observed, b.forecast]); lo, hi = allv.min(), allv.max(); pad = 0.03 * (hi - lo)
    for i, (m, c) in enumerate(MODELS):
        g = b[b.model == m].sort_values("date"); o, f = g.observed.values, g.forecast.values
        ax = fig.add_subplot(gs[i, 0])
        ax.plot(g.date, o, color="#222", lw=0.8); ax.plot(g.date, f, color=c, lw=0.8); ax.plot(per.date, per.forecast, color="#888", lw=0.6, ls=":")
        ax.set_ylim(lo - pad, hi + pad); ax.set_ylabel(name if tg == "GWETROOT" else "SPEI-1", fontsize=8)
        ax.set_title(f"{LAB.get(m, m)}  ($h$ = 14 d)", fontsize=8.5, loc="left")
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        ax.text(-0.1, 1.08, f"({'aceg'[i]})", transform=ax.transAxes, weight="bold", fontsize=9)
        ax2 = fig.add_subplot(gs[i, 1])
        ax2.scatter(o, f, s=3, color=c, alpha=0.3, edgecolors="none")
        ax2.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#444", lw=0.8, ls="--")
        ax2.set_xlim(lo - pad, hi + pad); ax2.set_ylim(lo - pad, hi + pad); ax2.set_aspect("equal")
        rmse = np.sqrt(np.mean((o - f) ** 2)); nse = 1 - np.sum((o - f) ** 2) / np.sum((o - o.mean()) ** 2)
        ax2.text(0.05, 0.95, f"RMSE = {rmse:.3f}\nNSE = {nse:.3f}", transform=ax2.transAxes, va="top", fontsize=6.8,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbb", lw=0.5))
        ax2.set_ylabel("Forecast", fontsize=8)
        if i == 3: ax2.set_xlabel("Observed", fontsize=8)
        for s in ("top", "right"): ax2.spines[s].set_visible(False)
        ax2.text(-0.42, 1.08, f"({'bdfh'[i]})", transform=ax2.transAxes, weight="bold", fontsize=9)
        print(tag, m, round(rmse, 3), round(nse, 3))
    fig.legend([Line2D([], [], color="#222", lw=1.2), Line2D([], [], color="#555", lw=1.2), Line2D([], [], color="#888", lw=1, ls=":")],
               ["Observed", "Model forecast (colour by model)", "Persistence"], loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.53, 0.995), fontsize=8)
    fig.savefig(f"{OUT}/fig{tag}_pred_vs_obs_{tg}.png", dpi=400); fig.savefig(f"{OUT}/fig{tag}_pred_vs_obs_{tg}.pdf"); plt.close(fig)
