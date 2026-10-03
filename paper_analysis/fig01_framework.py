"""Figure 1: methodological framework, with the ten workflow stages of Section 2.1 marked (i)-(x)."""
import os
from _paths import OUT as _OUT
os.environ.setdefault("FIG_OUT", str(_OUT))
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import make_all_figures_pro as F
from make_all_figures_pro import rbox, arrow, NAVY, PBLU, BLU, PNEU, INK, MUT, SAND, ACC, PACC, PTEAL, TEAL
OUT = os.environ["FIG_OUT"]; os.makedirs(OUT, exist_ok=True)

def fig1_stages():
    fig, ax = plt.subplots(figsize=(7.8, 10.47)); ax.set_xlim(0, 10); ax.set_ylim(-0.56, 13.35); ax.axis("off")
    L, R = 1.0, 9.0; W = R - L; CX = 5.0
    def phase(y, txt, c):
        ax.add_patch(Rectangle((0.18, y - 0.34), 0.12, 0.68, facecolor=c, edgecolor="none", zorder=3))
        ax.text(0.46, y, txt, rotation=90, ha="center", va="center", fontsize=7.4, color=c, weight="bold")
    def badge(x, y, lab):
        ax.add_patch(Circle((x, y), 0.2, facecolor=INK, edgecolor="white", lw=0.8, zorder=6))
        ax.text(x, y, lab, ha="center", va="center", fontsize=6.6, color="white", weight="bold", zorder=7)
    BX = 9.42
    rbox(ax, L, 11.86, W, 1.12,
         "Daily NASA POWER reanalysis  (1996\u20132025,  n = 10,958)\n"
         "PRECTOTCORR \u00b7 T2M \u00b7 RH2M \u00b7 ALLSKY_SFC_SW_DWN \u00b7 WS2M \u00b7 GWETROOT\n"
         "quality control and gap screening",
         fc=NAVY, ec=NAVY, tc="white", fs=7.5, weight="bold"); phase(12.42, "DATA", NAVY)
    badge(BX, 12.62, "i"); badge(BX, 12.1, "ii")
    ax.text(BX, 12.95, "Step", ha="center", va="bottom", fontsize=7.2, color=INK, weight="bold")
    arrow(ax, CX, 11.86, CX, 11.62)
    rbox(ax, L, 10.66, W, 0.96,
         "PET (FAO-56 Penman\u2013Monteith)  \u2192  30-day water balance  D = P \u2212 PET\n"
         "Standardized indices: SPEI and SSI   (calibrated without using the test period)",
         fc=PBLU, ec=BLU, fs=8.0); phase(11.14, "INDICES", BLU)
    badge(BX, 11.36, "iii"); badge(BX, 10.9, "iv")
    arrow(ax, CX, 10.66, CX, 10.28)
    rbox(ax, L, 9.1, W, 1.18,
         "Supervised reformulation:  sliding 90-day window  \u2192  target at lead h\n"
         "Chronological split:  calibration 1996\u20132020 (early stopping on its final 10%)\n"
         "test 2021\u20132025",
         fc=PNEU, ec=INK, fs=7.9); phase(9.69, "SETUP", MUT); badge(BX, 9.69, "v")
    arrow(ax, CX, 9.1, CX, 8.72)
    rbox(ax, L, 7.9, W, 0.82,
         "Forecasting models (3\u20135 seeds):  TCN \u00b7 LSTM \u00b7 Transformer \u00b7 Random Forest\n"
         "Reference baselines:  persistence  and  day-of-year climatology",
         fc=PNEU, ec=INK, fs=7.9); phase(8.31, "MODELS", MUT)
    arrow(ax, CX, 7.9, CX, 7.52)
    rbox(ax, 2.2, 6.6, 5.6, 0.74,
         "Multi-horizon forecasts   h = 1, 7, 14, 30 days\nGWETROOT  and  rolling 30-day SPEI", fc=SAND, ec=ACC, fs=8.0)
    phase(6.97, "FORECASTS", ACC)
    ax.plot([9.12, 9.2, 9.2, 9.12], [8.31, 8.31, 6.97, 6.97], color=INK, lw=0.9)          # stage vi spans models + forecasts
    ax.plot([7.8, 9.12], [6.97, 6.97], color=INK, lw=0.6, ls=":")
    badge(BX, 7.64, "vi")
    arrow(ax, 3.4, 6.6, 2.85, 6, rad=0.12); arrow(ax, 6.6, 6.6, 7.15, 6, rad=-0.12)
    rbox(ax, 0.6, 4.74, 4.0, 1.26,
         "Continuous skill\nRMSE \u00b7 MAE \u00b7 R\u00b2 \u00b7 NSE \u00b7 KGE\nskill score vs baselines\nDiebold\u2013Mariano test", fc=PBLU, ec=BLU, fs=7.8)
    rbox(ax, 5.4, 4.74, 4.0, 1.26,
         "Warning skill\ndrought alerts (SPEI \u2227/\u2228 SSI, D0\u2013D3)\nPOD \u00b7 FAR \u00b7 CSI \u00b7 frequency bias\nevent-based verification", fc=PACC, ec=ACC, fs=7.8)
    phase(5.37, "EVALUATION", INK); badge(4.6, 6, "vii"); badge(9.4, 6, "viii")
    arrow(ax, 2.6, 4.74, 4.2, 3.98, rad=-0.10); arrow(ax, 7.4, 4.74, 5.8, 3.98, rad=0.10)
    rbox(ax, 1.7, 2.82, 6.6, 0.96,
         "Forecast skill is not warning skill\nopen, transferable framework + decision support (DSS)",
         fc=NAVY, ec=NAVY, tc="white", fs=8.4, weight="bold")
    arrow(ax, CX, 2.82, CX, 2.44)
    rbox(ax, 1.7, 1.54, 6.6, 0.9,
         "Predictor ablation  \u00b7  spatial transfer (leave-one-cell-out)\n"
         "robustness and sensitivity analyses", fc=PTEAL, ec=TEAL, fs=7.9)
    phase(1.99, "ANALYSES", TEAL); badge(8.62, 1.99, "ix")
    arrow(ax, CX, 1.54, CX, 1.16)
    rbox(ax, 1.7, 0.36, 6.6, 0.8, "Open release: code, configuration and data-access scripts", fc=PNEU, ec=INK, fs=7.9)
    phase(0.76, "RELEASE", MUT); badge(8.62, 0.76, "x")
    fig.savefig(f"{OUT}/fig01_framework.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{OUT}/fig01_framework.pdf", bbox_inches="tight"); plt.close(fig); print("ok")

if __name__ == "__main__":
    fig1_stages()
