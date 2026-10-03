"""Figure 11: performance (Roebber) diagram for moderate-drought alerts (OR rule, level D1), single cell. Success ratio
(1 - FAR) against probability of detection, with CSI contours and dashed lines of constant frequency bias; each model's
path joins its four lead times. Input: results/dss_validation.csv."""
import numpy as np, pandas as pd, matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from _paths import RES, OUT
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 7.8, "axes.linewidth": 0.8})
MOD = [("TCN", "#0072B2", "o"), ("LSTM", "#009E73", "^"), ("Transformer", "#CC79A7", "D"), ("RandomForest", "#E69F00", "s"),
       ("Persistence", "#111111", "X")]
LAB = {"RandomForest": "Random forest"}
v = pd.read_csv(RES / "dss_validation.csv"); v = v[(v.logic == "or") & (v.level == "D1")]
fig = plt.figure(figsize=(5.37, 5.2)); ax = fig.add_axes([0.09, 0.16, 0.74, 0.79]); cax = fig.add_axes([0.855, 0.16, 0.035, 0.79])
sr, pod = np.meshgrid(np.linspace(0.001, 1, 400), np.linspace(0.001, 1, 400)); csi = 1 / (1 / sr + 1 / pod - 1)
cf = ax.contourf(sr, pod, csi, levels=np.arange(0, 1.01, 0.1), cmap="GnBu", alpha=0.5)
cb = fig.colorbar(cf, cax=cax, ticks=np.arange(0, 1.01, 0.2)); cb.set_label("CSI", labelpad=3); cb.outline.set_linewidth(0.8)
cl = ax.contour(sr, pod, csi, levels=np.arange(0.1, 1.0, 0.1), colors="#777", linewidths=0.6)
ax.clabel(cl, fmt="%.1f", fontsize=5.8, inline=True)
for fb in (0.5, 1, 1.5, 2, 4, 10):
    x = np.linspace(0, 1, 50); ax.plot(x, fb * x, color="#999", lw=0.7, ls=(0, (3, 2)), zorder=1)
    lx, ly = (0.92 / fb, 0.92) if fb >= 1 else (0.93, 0.93 * fb)
    if fb < 10: ax.text(lx, ly, f"{fb:g}", fontsize=6.2, color="#777", ha="center", va="center", bbox=dict(fc="white", ec="none", pad=0.4, alpha=0.7))
placed = []
def put(x, y):
    for _ in range(12):
        if all(abs(x - a) > 0.03 or abs(y - b) > 0.019 for a, b in placed): break
        y -= 0.02
    placed.append((x, y)); return x, y
for m, c, mk in MOD:
    g = v[v.model == m].set_index("horizon").reindex([1, 7, 14, 30])
    x = (1 - g.FAR).fillna(0).values; y = g.POD.values
    ax.plot(x, y, color=c if m != "Persistence" else "#333", lw=1.2, alpha=0.75, zorder=3)
    ax.scatter(x, y, marker=mk, s=34 if m != "Persistence" else 42, color=c, edgecolors="white", linewidths=0.6, zorder=4, clip_on=False)
    for h, xi, yi in zip([1, 7, 14, 30], x, y):
        tx, ty = put(xi + 0.012, yi + 0.008); ax.text(tx, ty, str(h), color=c, fontsize=6.6, zorder=5)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xlabel("Success ratio (1 - FAR)"); ax.set_ylabel("Probability of detection")
ax.set_title("DSS performance diagram (OR logic, alert \u2265 D1)", fontsize=10)
ax.legend([Line2D([], [], ls="none", marker=mk, ms=5.5, mfc=c, mec="white") for _, c, mk in MOD], [LAB.get(m, m) for m, _, _ in MOD],
          loc="upper left", ncol=2, frameon=False, fontsize=7, handletextpad=0.3, columnspacing=1.2)
fig.text(0.005, 0.01, "Numbers = lead time (days).", fontsize=6.8, color="#666")
fig.savefig(OUT / "fig11_performance_diagram.png", dpi=600, bbox_inches="tight", pad_inches=0.04); fig.savefig(OUT / "fig11_performance_diagram.pdf", bbox_inches="tight", pad_inches=0.04); print("ok")
