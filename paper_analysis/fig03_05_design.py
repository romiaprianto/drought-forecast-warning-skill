"""Figures 3 and 5: leakage-prevention workflow and experimental design, as implemented in the released code
(calibration 1996-2020 with early stopping on its final 10 % of windows; test 2021-2025; no hyperparameter tuning)."""
import matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrowPatch
from _paths import OUT
plt.rcParams.update({"font.family": "DejaVu Serif"})
CAL, ES, TEST, DARK, BOX, NAVY = "#1F4E79", "#2E8B7A", "#C0762F", "#222833", "#F0F4F8", "#1F4E79"

def bar(ax, x0, x1, y, h, segs, fs=11):
    for a, b, col, lab in segs:
        ax.add_patch(Rectangle((a, y), b - a, h, fc=col, ec="white", lw=1.5))
        if lab: ax.text((a + b) / 2, y + h / 2, lab, ha="center", va="center", color="white", fontsize=fs, weight="bold")

# ------------------------------------------------------------------ Figure 3
fig, ax = plt.subplots(figsize=(9.5, 9.0)); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
ax.text(55, 97, "Chronological split  (no shuffling)", ha="center", fontsize=13, weight="bold")
x0, x1 = 14, 94; frac = (2020.5 - 1996) / (2025.99 - 1996); xs = x0 + frac * (x1 - x0); xes = xs - 0.10 * (xs - x0)
bar(ax, x0, x1, 88, 6, [(x0, xes, CAL, "CALIBRATION  1996\u20132020"), (xes, xs, ES, ""), (xs, x1, TEST, "TEST")], 11)
ax.text((xes + xs) / 2, 91, "ES", ha="center", va="center", color="white", fontsize=9, weight="bold")
steps = [("Split the record chronologically into CALIBRATION (1996\u20132020)\nand TEST (2021\u20132025)", [CAL, TEST]),
         ("Estimate on CALIBRATION only: SPEI and SSI day-of-year\nreference samples (empirical CDF, \u00b115 days) and Min\u2013Max bounds", [CAL]),
         ("Transform TEST with the calibration-derived statistics", [TEST]),
         ("Train each model once on CALIBRATION; early stopping on the\nfinal 10% of calibration windows (ES); no hyperparameter tuning", [CAL, ES]),
         ("Evaluate once on TEST", [TEST])]
ys = [76, 62, 48, 34, 20]
for k, ((txt, cols), y) in enumerate(zip(steps, ys), 1):
    ax.add_patch(Circle((6, y), 3.4, fc=DARK)); ax.text(6, y, str(k), ha="center", va="center", color="white", fontsize=14, weight="bold")
    ax.add_patch(FancyBboxPatch((14, y - 5), 80, 10, boxstyle="round,pad=0.2,rounding_size=0.8", fc=BOX, ec=DARK, lw=1.4))
    for j, c in enumerate(reversed(cols)):
        hh = 7.0 / len(cols); ax.add_patch(Rectangle((15.5, y - 3.5 + j * hh), 2.3, hh, fc=c, ec="none"))
    ax.text(21, y, txt, va="center", fontsize=11.2, color=DARK)
    top = 88 if k == 1 else ys[k - 2] - 5
    ax.add_patch(FancyArrowPatch((54, top - 0.3), (54, y + 5.3), arrowstyle="-|>", mutation_scale=16, lw=2, color=DARK))
for xx, c, lab in [(16, CAL, "CALIBRATION"), (43, ES, "EARLY STOPPING (ES)"), (76, TEST, "TEST")]:
    ax.add_patch(Rectangle((xx, 9.2), 3, 2.2, fc=c)); ax.text(xx + 4, 10.3, lab, va="center", fontsize=10.5)
ax.text(55, 4.5, "No statistic or model parameter is derived from the test period; hyperparameters are fixed in advance.",
        ha="center", fontsize=10.5, style="italic", color=NAVY)
fig.savefig(OUT / "fig03_leakage_workflow.png", dpi=300, bbox_inches="tight"); fig.savefig(OUT / "fig03_leakage_workflow.pdf", bbox_inches="tight")

# ------------------------------------------------------------------ Figure 5
fig, ax = plt.subplots(figsize=(11.7, 9.3)); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
x0, x1 = 3, 97; X = lambda yr: x0 + (yr - 1996) / (2026 - 1996) * (x1 - x0); xes = X(2018.5)
bar(ax, x0, x1, 88, 7, [(x0, xes, CAL, "CALIBRATION  1996\u20132020"), (xes, X(2021), ES, ""), (X(2021), x1, TEST, "TEST\n2021\u201325")], 12)
ax.text((xes + X(2021)) / 2, 91.5, "early\nstopping", ha="center", va="center", color="white", fontsize=8.5, weight="bold")
for yr, ha in ((1996, "center"), (2018.5, "right"), (2021, "left"), (2026, "center")):
    ax.text(X(yr) + (-0.4 if ha == "right" else 0.4 if ha == "left" else 0), 96.5,
            {2018.5: "\u2248 mid-2018", 2026: "2025"}.get(yr, str(int(yr))), ha=ha, fontsize=10.5)
ax.text(50, 81, "Indices and scalers calibrated on 1996\u20132020; each model fitted once, with early stopping on the final 10% of\n"
        "calibration windows; hyperparameters fixed in advance; TEST used once.", ha="center", va="center", fontsize=11, style="italic", color=NAVY)
ax.text(3, 71.5, "Sliding-window construction", fontsize=14, weight="bold")
ax.add_patch(FancyBboxPatch((6, 50), 29, 9, boxstyle="round,pad=0.2,rounding_size=0.8", fc="#DAE3ED", ec=DARK, lw=1.6))
ax.text(20.5, 54.5, "input window\nL = 90 days", ha="center", va="center", fontsize=13)
for i, (h, y) in enumerate(zip(["h = 1 d", "h = 7 d", "h = 14 d", "h = 30 d"], [64, 57, 50, 43])):
    ax.add_patch(FancyBboxPatch((56, y - 2.4), 16, 4.8, boxstyle="round,pad=0.2,rounding_size=0.6", fc="#EAF1F8", ec="#2F6FA7", lw=1.6))
    ax.text(64, y, h, ha="center", va="center", fontsize=12.5, color="#2F6FA7")
    ax.add_patch(FancyArrowPatch((35.3, 54.5), (55.6, y), connectionstyle=f"arc3,rad={0.18 - i * 0.12}", arrowstyle="-|>",
                                 mutation_scale=16, lw=1.8, color="#2F6FA7"))
ax.text(75, 54.5, "one model per lead,\njoint output for\nGWETROOT and SPEI", va="center", fontsize=10.5, color=DARK)
ax.text(75, 45.5, "\u00d7 5 seeds (single cell)\n\u00d7 3 seeds (multi-cell)", va="center", fontsize=11, color=TEST, style="italic")
ax.text(3, 33, "Spatial transfer: leave-one-cell-out", fontsize=14, weight="bold")
ax.text(3, 27.5, "Train on 9 cells  \u2192  test on the withheld cell, which keeps its own record (not an ungauged test)", fontsize=12)
for i in range(10):
    xx = 4 + i * 9.4; held = i == 4
    ax.add_patch(FancyBboxPatch((xx, 16), 7, 6, boxstyle="round,pad=0.2,rounding_size=0.6", fc=TEST if held else "#DAE3ED", ec=DARK, lw=1.5))
    ax.text(xx + 3.5, 19, "test" if held else "train", ha="center", va="center", fontsize=11, color="white" if held else DARK)
ax.text(7.5, 12.5, "cell 1", ha="center", fontsize=10.5); ax.text(4 + 9 * 9.4 + 3.5, 12.5, "cell 10", ha="center", fontsize=10.5)
ax.text(50, 4, "Static covariates are z-scored on the training cells of each fold; the withheld cell's inputs and targets are\n"
        "Min\u2013Max scaled with its own 1996\u20132020 record, and its lagged soil moisture enters the input window.",
        ha="center", va="center", fontsize=10.5, style="italic", color=NAVY)
fig.savefig(OUT / "fig05_experimental_design.png", dpi=300, bbox_inches="tight"); fig.savefig(OUT / "fig05_experimental_design.pdf", bbox_inches="tight")
print("ok")
