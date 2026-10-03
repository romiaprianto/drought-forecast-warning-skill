"""Figure 8 (version sent to the mentor): TCN forecast against observation at the 14-day lead, single cell.
Same content and layout as the published panel set; the legend sits above the time-series panels so it cannot
overlap the data. Input: results/predictions.csv."""
import sys, numpy as np, pandas as pd, matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from _paths import RES, OUT as _OUT
PRED = RES / "predictions.csv"; OUT = str(_OUT / "fig08_pred_vs_obs")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.linewidth": 0.8, "mathtext.default": "it"})
pr = pd.read_csv(PRED, parse_dates=["date"]); h = 14; BLUE = "#1f77b4"
fig = plt.figure(figsize=(7.4, 4.75)); gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], hspace=0.42, wspace=0.28,
                                                            left=0.08, right=0.99, top=0.86, bottom=0.09)
stats = {}
for j, (tg, name, ylab) in enumerate([("GWETROOT", "Root-zone soil moisture", "Root-zone soil moisture"), ("SPEI_1", "SPEI-1", "SPEI-1")]):
    b = pr[(pr.target == tg) & (pr.horizon == h)]
    obs = b[b.model == "TCN"].sort_values("date"); per = b[b.model == "Persistence"].sort_values("date")
    ax = fig.add_subplot(gs[0, j])
    ax.plot(obs.date, obs.observed, color="#222", lw=0.9)
    ax.plot(obs.date, obs.forecast, color=BLUE, lw=0.9)
    ax.plot(per.date, per.forecast, color="#888", lw=0.7, ls=":")
    ax.set_title(f"{name}  ($h$ = 14 d)", fontsize=9); ax.set_ylabel(ylab)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.text(-0.13, 1.08, "(a)" if j == 0 else "(c)", transform=ax.transAxes, weight="bold", fontsize=9.5)
    ax2 = fig.add_subplot(gs[1, j]); o, f = obs.observed.values, obs.forecast.values
    ax2.scatter(o, f, s=5, color=BLUE, alpha=0.3, edgecolors="none")
    lo, hi = min(o.min(), f.min()), max(o.max(), f.max()); pad = 0.03 * (hi - lo)
    ax2.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#444", lw=0.9, ls="--")
    ax2.set_xlim(lo - pad, hi + pad); ax2.set_ylim(lo - pad, hi + pad); ax2.set_aspect("equal")
    rmse = float(np.sqrt(np.mean((o - f) ** 2))); nse = 1 - np.sum((o - f) ** 2) / np.sum((o - o.mean()) ** 2); stats[tg] = (rmse, nse)
    ax2.text(0.05, 0.95, f"RMSE = {rmse:.3f}\nNSE = {nse:.3f}", transform=ax2.transAxes, va="top", fontsize=7.5,
             bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#bbb", lw=0.6))
    ax2.set_xlabel("Observed"); ax2.set_ylabel("Forecast (TCN)")
    for s in ("top", "right"): ax2.spines[s].set_visible(False)
    ax2.text(-0.32, 1.06, "(b)" if j == 0 else "(d)", transform=ax2.transAxes, weight="bold", fontsize=9.5)
fig.legend([Line2D([], [], color="#222", lw=1.2), Line2D([], [], color=BLUE, lw=1.2), Line2D([], [], color="#888", lw=1, ls=":")],
           ["Observed", "TCN", "Persistence"], loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.53, 0.995), fontsize=8.5)
fig.savefig(OUT + ".png", dpi=600); fig.savefig(OUT + ".pdf")
print({k: (round(v[0], 3), round(v[1], 3)) for k, v in stats.items()})
