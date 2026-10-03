"""Unified, professional regeneration of all six Methods figures (figure1..figure6)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Circle, Ellipse, Polygon
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection

# ---------------- unified design system ----------------
mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9,
    "axes.linewidth": 0.8, "savefig.dpi": 300, "savefig.bbox": "tight", "mathtext.fontset": "cm",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
INK   = "#1f2632"   # near-black slate
NAVY  = "#1f4e79"   # deep blue  (data / TRAIN / structure)
BLU   = "#2f6fa6"   # medium blue
TEAL  = "#2f8f7f"   # teal       (VALIDATION)
ACC   = "#c1742d"   # warm amber (TEST / warning / highlight)
PBLU  = "#e9f1f8"; PTEAL = "#e6f2ef"; PACC = "#f8eee2"; PNEU = "#f3f6f9"; SAND = "#f6efe4"
GRID  = "#dde3e9"; MUT = "#6b7480"; EDGE = "#cfd6de"
import os
OUT = os.environ.get("FIG_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper_outputs"))
WET2DRY = LinearSegmentedColormap.from_list("wet2dry", ["#2f8f7f", "#8fb98a", "#e7c873", "#c1742d", "#9c4f1d"])

def rbox(ax, x, y, w, h, text, fc=PNEU, ec=INK, fs=8.4, weight="normal", tc=INK, lw=1.0, ha="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.035",
                 linewidth=lw, edgecolor=ec, facecolor=fc, zorder=3))
    tx = x + w/2 if ha == "center" else x + 0.18
    ax.text(tx, y + h/2, text, ha=ha, va="center", fontsize=fs, color=tc, weight=weight,
            zorder=4, linespacing=1.34)

def arrow(ax, x1, y1, x2, y2, ec=INK, lw=1.2, rad=0.0, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=13,
                 lw=lw, color=ec, connectionstyle=f"arc3,rad={rad}", zorder=2))

def badge(ax, n, x, y, r=0.30, fc=INK):
    ax.add_patch(Circle((x, y), r, facecolor=fc, edgecolor="none", zorder=6))
    ax.text(x, y, str(n), color="white", ha="center", va="center", fontsize=9.5, weight="bold", zorder=7)


# =================== FIGURE 1 — FRAMEWORK ===================
def fig1_framework():
    fig, ax = plt.subplots(figsize=(7.8, 9.4)); ax.set_xlim(0, 10); ax.set_ylim(0, 13.0); ax.axis("off")
    L, R = 1.0, 9.0; W = R - L; CX = 5.0
    # phase rail labels (left)
    def phase(y, txt, c):
        ax.add_patch(Rectangle((0.18, y-0.34), 0.12, 0.68, facecolor=c, edgecolor="none", zorder=3))
        ax.text(0.46, y, txt, rotation=90, ha="center", va="center", fontsize=7.4, color=c, weight="bold")
    # 1 data
    rbox(ax, L, 12.0, W, 0.86,
         "Daily NASA POWER reanalysis  (1996\u20132025,  n = 10,958)\n"
         "PRECTOTCORR \u00b7 T2M \u00b7 RH2M \u00b7 ALLSKY_SFC_SW_DWN \u00b7 WS2M \u00b7 GWETROOT",
         fc=NAVY, ec=NAVY, tc="white", fs=7.7, weight="bold"); phase(12.43, "DATA", NAVY)
    arrow(ax, CX, 12.0, CX, 11.62)
    # 2 indices
    rbox(ax, L, 10.66, W, 0.96,
         "PET (FAO-56 Penman\u2013Monteith)  \u2192  30-day water balance  D = P \u2212 PET\n"
         "Standardized indices: SPEI and SSI   (calibrated without using the test period)",
         fc=PBLU, ec=BLU, fs=8.0); phase(11.14, "INDICES", BLU)
    arrow(ax, CX, 10.66, CX, 10.28)
    # 3 supervised
    rbox(ax, L, 9.46, W, 0.82,
         "Sliding 90-day window  \u2192  target at lead h   (ten cells pooled)\n"
         "Chronological split   calibration 1996\u20132020  |  test 2021\u20132025",
         fc=PNEU, ec=INK, fs=7.9); phase(9.87, "SETUP", MUT)
    arrow(ax, CX, 9.46, CX, 9.08)
    # 4 models
    rbox(ax, L, 8.26, W, 0.82,
         "Forecasting models:  TCN \u00b7 LSTM \u00b7 Transformer \u00b7 Random Forest\n"
         "Reference baselines:  persistence  and  day-of-year climatology",
         fc=PNEU, ec=INK, fs=7.9); phase(8.67, "MODELS", MUT)
    arrow(ax, CX, 8.26, CX, 7.88)
    # 5 multi-horizon (centered, narrower)
    rbox(ax, 2.2, 6.96, 5.6, 0.74,
         "Multi-horizon forecasts   h = 1, 7, 14, 30 days\nGWETROOT  and  rolling 30-day SPEI", fc=SAND, ec=ACC, fs=8.0)
    phase(7.33, "FORECASTS", ACC)
    # symmetric split to two evaluation branches
    arrow(ax, 3.4, 6.96, 2.85, 6.36, rad=0.12); arrow(ax, 6.6, 6.96, 7.15, 6.36, rad=-0.12)
    rbox(ax, 0.6, 5.1, 4.0, 1.26,
         "Continuous skill\nRMSE \u00b7 MAE \u00b7 R\u00b2 \u00b7 NSE \u00b7 KGE\nskill score vs baselines\nmoving-block bootstrap intervals", fc=PBLU, ec=BLU, fs=7.8)
    rbox(ax, 5.4, 5.1, 4.0, 1.26,
         "Warning skill\nraw and calibrated alerts (SPEI \u2227/\u2228 SSI)\nPOD \u00b7 FAR \u00b7 CSI \u00b7 bias \u00b7 PSS \u00b7 ETS \u00b7 SEDI\nvs index-space persistence", fc=PACC, ec=ACC, fs=7.8)
    phase(5.73, "EVALUATION", INK)
    # converge
    arrow(ax, 2.6, 5.1, 4.2, 4.34, rad=-0.10); arrow(ax, 7.4, 5.1, 5.8, 4.34, rad=0.10)
    rbox(ax, 1.7, 3.18, 6.6, 0.96,
         "Forecast skill is not warning skill\nopen, transferable framework + decision support (DSS)",
         fc=NAVY, ec=NAVY, tc="white", fs=8.4, weight="bold")
    arrow(ax, CX, 3.18, CX, 2.8)
    # spatial transfer footer
    rbox(ax, 1.7, 1.9, 6.6, 0.72,
         "Moving-block bootstrap: 60-day blocks, 400 resamples\n(95% intervals for scores and model\u2013persistence differences)", fc=PTEAL, ec=TEAL, fs=7.9)
    phase(2.26, "UNCERTAINTY", TEAL)
    for ext, kw in (("png", {}), ("pdf", {}), ("eps", {}), ("tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}})):
        fig.savefig(f"{OUT}/figure1_framework.{ext}", **kw)
    plt.close(fig); print("fig1 framework (png, pdf, eps, tiff)")


# =================== FIGURE 2 — STUDY AREA (polished schematic locator) ===================
def fig2_studyarea():
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    lon0, lon1, lat0, lat1 = 115.4, 125.4, -10.9, -7.7
    ax.set_xlim(lon0, lon1); ax.set_ylim(lat0, lat1)
    # ocean
    ax.add_patch(Rectangle((lon0, lat0), lon1-lon0, lat1-lat0, facecolor="#dceaf2", edgecolor="none", zorder=0))
    # graticule
    for xv in range(116, 126, 2): ax.axvline(xv, color="#c4d6e0", lw=0.6, zorder=1)
    for yv in range(-10, -7): ax.axhline(yv, color="#c4d6e0", lw=0.6, zorder=1)
    # stylized island outlines (schematic ellipses) : (lon, lat, w, h, angle, label, lx, ly)
    islands = [
        (116.35, -8.62, 0.60, 0.55, 0,  "Lombok",  116.35, -8.18),
        (117.75, -8.66, 2.05, 0.62, -7, "Sumbawa", 117.7,  -8.16),
        (121.55, -8.60, 3.10, 0.55, -4, "Flores",  121.7,  -8.13),
        (120.00, -9.72, 1.45, 0.52, -10,"Sumba",   119.7,  -10.18),
        (123.95, -9.98, 1.95, 0.78, -38,"Timor",   124.35, -10.42),
    ]
    for lon, lat, w, h, ang, lab, lx, ly in islands:
        ax.add_patch(Ellipse((lon, lat), w, h, angle=ang, facecolor="#e9e2d2",
                     edgecolor="#b9ad93", lw=1.0, zorder=2))
        ax.text(lx, ly, lab, fontsize=7.6, style="italic", color="#5d5443", ha="center", va="center", zorder=5)
    # cells (Table 1), colored by mean dry-season (JJAS) precipitation
    cells = [("C01",116.3,-8.6,137),("C02",116.9,-8.8,111),("C03",117.4,-8.5,72),("C04",118.5,-8.5,71),
             ("C05",120.4,-8.6,150),("C06",121.6,-8.7,83),("C07",122.9,-8.4,64),("C08",120.2,-9.6,84),
             ("C09",123.6,-10.1,37),("C10",124.2,-9.9,34)]
    lon=np.array([c[1] for c in cells]); lat=np.array([c[2] for c in cells]); dvals=np.array([c[3] for c in cells])
    sc=ax.scatter(lon, lat, c=dvals, cmap=WET2DRY.reversed(), vmin=30, vmax=155, s=130, edgecolor="white", linewidth=1.2, zorder=6)
    for cid,lo,la,_ in cells:
        ax.annotate(cid,(lo,la),textcoords="offset points",xytext=(0,7.5),ha="center",
                    fontsize=6.6,color=INK,weight="bold",zorder=7)
    # colorbar
    cb=fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.015); cb.set_label("dry-season precipitation\n(June\u2013September, mm)", fontsize=8)
    cb.set_ticks([40,80,120,150]); cb.ax.tick_params(labelsize=7)
    # north arrow (empty NW corner)
    nx=115.75
    ax.annotate("", xy=(nx,-7.82), xytext=(nx,-8.34), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4), zorder=8)
    ax.text(nx,-7.74,"N", ha="center", va="bottom", fontsize=9, weight="bold", zorder=8)
    # inset: Indonesia context (schematic)
    iax=fig.add_axes([0.10,0.16,0.235,0.245]); iax.set_xlim(94,142); iax.set_ylim(-11,7); iax.axis("off")
    iax.add_patch(Rectangle((94,-11),48,18,facecolor="#dceaf2",edgecolor="#bcd0db",lw=0.8))
    for lon,lat,w,h,a in [(101,0,9,4,18),(110,-7.2,8,1.8,-5),(114,0.5,7,7,0),
                          (121,0,4,8,10),(133,-4,12,7,0),(120.5,-9,9,1.4,-8)]:
        iax.add_patch(Ellipse((lon,lat),w,h,angle=a,facecolor="#e3ddcd",edgecolor="#b9ad93",lw=0.6))
    iax.add_patch(Rectangle((115.4,-10.9),10,3.2,facecolor="none",edgecolor="#c0392b",lw=1.6))
    iax.text(118,5,"Indonesia",fontsize=7,style="italic",color=MUT,ha="center")
    ax.set_xlabel("longitude (\u00b0E)", fontsize=8.5); ax.set_ylabel("latitude (\u00b0N)", fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    ax.text(lon1-0.1, lat0+0.12, "schematic locator; island outlines stylized, coordinates from Table 1",
            ha="right", va="bottom", fontsize=6.6, style="italic", color=MUT, zorder=8)
    fig.savefig(f"{OUT}/figure2_studyarea.png"); plt.close(fig); print("fig2 studyarea")


# =================== FIGURE 3 — LEAKAGE FLOWCHART ===================
def fig3_leakage():
    fig, ax = plt.subplots(figsize=(7.6, 7.5)); ax.set_xlim(0, 10); ax.set_ylim(2.2, 12.3); ax.axis("off")
    BX0, BX1 = 2.0, 8.7; BW = BX1 - BX0; CX = (BX0+BX1)/2
    def vd(y1, y2): arrow(ax, CX, y1, CX, y2, lw=1.3)
    def opbox(y0, h, text, tabs):
        ax.add_patch(FancyBboxPatch((BX0, y0), BW, h, boxstyle="round,pad=0.012,rounding_size=0.04",
                     lw=1.0, edgecolor=INK, facecolor=PNEU, zorder=3))
        tx, tw, pad = BX0+0.13, 0.22, 0.10; seg=(h-2*pad)/len(tabs)
        for i,c in enumerate(reversed(tabs)):
            ax.add_patch(Rectangle((tx, y0+pad+i*seg), tw, seg, facecolor=c, edgecolor="none", zorder=4))
        ax.text(BX0+0.64, y0+h/2, text, ha="left", va="center", fontsize=8.4, color=INK, zorder=4, linespacing=1.34)
    ax.text(CX, 12.05, "Chronological split  (no shuffling)", ha="center", fontsize=8.8, weight="bold")
    by0, bh = 11.32, 0.54; segs=[("TRAIN",21,NAVY),("VALID",4,TEAL),("TEST",5,ACC)]
    tot=sum(s[1] for s in segs); x=BX0
    for lab,w,c in segs:
        ww=BW*w/tot; ax.add_patch(Rectangle((x,by0),ww,bh,facecolor=c,edgecolor="white",lw=1.0,zorder=4))
        ax.text(x+ww/2,by0+bh/2,lab,ha="center",va="center",color="white",fontsize=7.6,weight="bold",zorder=5); x+=ww
    badge(ax,1,1.15,by0+bh/2)
    steps=[(0.86,"Estimate on TRAIN only:\nSPEI log-logistic parameters, SSI climatology,\nand Min\u2013Max scaling bounds",[NAVY]),
           (0.84,"Transform VALIDATION and TEST\nusing TRAIN-derived statistics",[TEAL,ACC]),
           (0.84,"Tune hyperparameters and apply early stopping\non VALIDATION",[TEAL]),
           (0.86,"Re-estimate preprocessing on TRAIN + VALIDATION,\nthen retrain with the selected configuration",[NAVY,TEAL]),
           (0.66,"Evaluate once on TEST",[ACC])]
    prev=by0; cur=by0-0.46
    for i,(h,t,tabs) in enumerate(steps,start=2):
        y0=cur-h; vd(prev,cur); opbox(y0,h,t,tabs); badge(ax,i,1.15,y0+h/2); prev=y0; cur=y0-0.46
    ly=prev-0.55; lx=BX0+0.2
    for lab,c in [("TRAIN",NAVY),("VALIDATION",TEAL),("TEST",ACC)]:
        ax.add_patch(Rectangle((lx,ly-0.10),0.30,0.20,facecolor=c,edgecolor="none")); ax.text(lx+0.40,ly,lab,fontsize=7.8,va="center"); lx+=2.25
    ax.text(CX, ly-0.62, "No parameter, statistic, or hyperparameter is derived from the test period.",
            ha="center", fontsize=8.2, color=NAVY, style="italic")
    fig.savefig(f"{OUT}/figure3_leakage.png"); plt.close(fig); print("fig3 leakage")


# =================== FIGURE 4 — ACF (targets, from data) ===================
def fig4_acf():
    from statsmodels.tsa.stattools import acf
    from scipy.stats import norm
    import tcn_drought_pipeline_v2 as P   # reuse the canonical loader + rolling-30-day day-of-year SPEI
    CSV = os.environ.get("BASE_CELL_CSV", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "cell_sumbawaC.csv"))
    df = P.load_data(CSV)                 # auto-detects ';' legacy vs ',' cell format
    df = P.calculate_pet(df, -8.5, 210.0, P.CONFIG["wind_var"], P.CONFIG["wind_default"])
    df = P.calculate_spei(df, "2016-12-31", P.CONFIG["spei_clip"],
                          P.CONFIG["spei_scale"], P.CONFIG["doy_halfwin"])
    G = df["GWETROOT"]; spei = df["SPEI_1"]
    def tr16(x): return x.loc[:"2016"].dropna().values
    fig,axs=plt.subplots(1,2,figsize=(7.8,3.2))
    for ax,(series,ttl,col) in zip(axs,[(tr16(G),"GWETROOT  (root-zone soil moisture)",BLU),
                                          (tr16(spei),"Rolling 30-day SPEI",TEAL)]):
        ac=acf(series,nlags=120,fft=True);lags=np.arange(len(ac));ci=1.96/np.sqrt(len(series))
        ax.axhspan(-ci,ci,color=GRID,alpha=0.9,zorder=0)
        ax.fill_between(lags,0,ac,color=col,alpha=0.22,zorder=1)
        ax.plot(lags,ac,color=col,lw=1.4,zorder=2)
        ax.axhline(0,color=INK,lw=0.7); ax.axvline(90,color=ACC,lw=1.3,ls=(0,(4,2)),zorder=3)
        ax.text(90,0.93," 90-day window",color=ACC,fontsize=7.4,ha="left",transform=ax.get_xaxis_transform())
        ax.set_title(ttl,fontsize=8.5,weight="bold",color=INK,pad=6)
        ax.set_xlabel("lag (days)",fontsize=8.2); ax.set_ylabel("autocorrelation",fontsize=8.2)
        ax.set_xlim(-2,120); ax.set_ylim(min(-0.12,ac.min()*1.1),1.05); ax.tick_params(labelsize=7.4)
        for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{OUT}/figure4_acf.png"); plt.close(fig); print("fig4 acf")


# =================== FIGURE 5 — EXPERIMENTAL DESIGN ===================
def fig5_experiment():
    fig, ax = plt.subplots(figsize=(8.2, 6.5)); ax.set_xlim(0, 10); ax.set_ylim(0, 8.7); ax.axis("off")
    # --- three-way chronological split bar ---
    x0, x1, y = 0.7, 9.3, 8.0; yrs=2025-1996+1
    s1=x0+(x1-x0)*(2016-1996+1)/yrs; s2=x0+(x1-x0)*(2020-1996+1)/yrs
    for xx,ww,c,lab,fs in [(x0,s1-x0,NAVY,"TRAIN  1996\u20132016",8.3),(s1,s2-s1,TEAL,"VALID\n2017\u201320",6.8),(s2,x1-s2,ACC,"TEST\n2021\u201325",7.2)]:
        ax.add_patch(FancyBboxPatch((xx,y-0.3),ww,0.6,boxstyle="round,pad=0.005,rounding_size=0.03",facecolor=c,edgecolor="white",lw=1.0,zorder=3))
        ax.text(xx+ww/2,y,lab,ha="center",va="center",color="white",fontsize=fs,weight="bold",zorder=4)
    for xx,lab in [(x0,"1996"),(s1,"2017"),(s2,"2021"),(x1,"2025")]: ax.text(xx,y+0.5,lab,fontsize=7.4,ha="center")
    ax.text(5.0,y-0.80,"Development: indices and scaler calibrated on TRAIN; hyperparameters tuned on VALIDATION.\nFinal model: preprocessing recalibrated and refit on TRAIN + VALIDATION, with TEST used once.",
            fontsize=7.4,color=NAVY,ha="center",va="center",style="italic",linespacing=1.3)
    # --- sliding window: input fans to four horizon targets (direct multi-horizon) ---
    wcy=5.05; ax.text(0.7,6.55,"Sliding-window construction",fontsize=8.7,weight="bold")
    ibx,ibw,ibh=0.9,2.7,0.86
    ax.add_patch(FancyBboxPatch((ibx,wcy-ibh/2),ibw,ibh,boxstyle="round,pad=0.006,rounding_size=0.05",facecolor="#d9e2ea",edgecolor=INK,lw=1.0,zorder=3))
    ax.text(ibx+ibw/2,wcy,"input window\nL = 90 days",ha="center",va="center",fontsize=8.2,zorder=4,linespacing=1.3)
    hbx,hbw,hbh=5.5,1.55,0.48; hys=[wcy+1.02,wcy+0.34,wcy-0.34,wcy-1.02]
    for hy,h in zip(hys,[1,7,14,30]):
        ax.add_patch(FancyBboxPatch((hbx,hy-hbh/2),hbw,hbh,boxstyle="round,pad=0.005,rounding_size=0.07",facecolor=PBLU,edgecolor=BLU,lw=0.9,zorder=3))
        ax.text(hbx+hbw/2,hy,f"h = {h} d",ha="center",va="center",fontsize=7.9,color=BLU,zorder=4)
        rad=0.0 if abs(hy-wcy)<0.5 else (-0.16 if hy>wcy else 0.16)
        arrow(ax,ibx+ibw,wcy,hbx,hy,ec=BLU,lw=1.0,rad=rad)
    ax.text(hbx+hbw+0.35,wcy,"\u00d7 5 seeds",ha="left",va="center",fontsize=8.2,color=ACC,style="italic")
    # --- spatial transfer: leave-one-cell-out (centered) ---
    ly=2.05; ax.text(0.7,ly+0.95,"Spatial transfer: leave-one-cell-out",fontsize=8.7,weight="bold")
    ax.text(0.7,ly+0.56,"Train on 9 cells  \u2192  test on the held-out cell  (mirrors an ungauged location)",fontsize=7.7)
    nb,bw,gap=10,0.64,0.22; tot=nb*bw+(nb-1)*gap; sx=(10-tot)/2
    for i in range(nb):
        cx=sx+i*(bw+gap); held=(i==4); c=ACC if held else "#d9e2ea"
        ax.add_patch(FancyBboxPatch((cx,ly-0.31),bw,0.6,boxstyle="round,pad=0.005,rounding_size=0.07",facecolor=c,edgecolor=INK,lw=0.9,zorder=3))
        ax.text(cx+bw/2,ly-0.01,("test" if held else "train"),ha="center",va="center",fontsize=6.5,color=("white" if held else INK),zorder=4)
    ax.text(sx+bw/2,ly-0.56,"cell 1",fontsize=6.3,ha="center"); ax.text(sx+(nb-1)*(bw+gap)+bw/2,ly-0.56,"cell 10",fontsize=6.3,ha="center")
    ax.text(5.0,ly-1.06,"Static covariates (latitude, longitude, elevation, mean P, mean dry-season P,\nmean soil moisture, aridity index) standardized on the training cells of each fold.",
            fontsize=7.1,color=NAVY,ha="center",va="center",style="italic",linespacing=1.3)
    fig.savefig(f"{OUT}/figure5_experimental_design.png"); plt.close(fig); print("fig5 experiment")


# =================== FIGURE 6 — TCN ARCHITECTURE ===================
def fig6_tcn():
    fig, ax = plt.subplots(figsize=(8.6, 3.9)); ax.set_xlim(0, 12.6); ax.set_ylim(0, 7.0); ax.axis("off")
    ax.text(6.3, 6.6, "Temporal Convolutional Network", ha="center", fontsize=10, weight="bold", color=INK)
    ax.text(6.3, 5.78, "dilated causal convolutions   (kernel size 3, two layers per block;  teal arcs = residual connections)",
            ha="center", fontsize=7.8, color=MUT)
    iw, bw, gap, dw, agap = 1.6, 1.3, 0.22, 1.55, 0.45
    total = iw + agap + (5*bw + 4*gap) + agap + dw
    sx = (12.6 - total)/2
    ybox, bh = 3.0, 1.6; cy = ybox + bh/2
    rbox(ax, sx, cy-0.65, iw, 1.3, "Input\n90 \u00d7 6\n(window \u00d7\nvariables)", fc=NAVY, ec=NAVY, tc="white", fs=7.6, weight="bold")
    arrow(ax, sx+iw, cy, sx+iw+agap, cy)
    bx0 = sx + iw + agap
    for i, d in enumerate([1, 2, 4, 8, 16]):
        x = bx0 + i*(bw+gap)
        rbox(ax, x, ybox, bw, bh, f"Residual\nblock\n\ndilation {d}", fc=PBLU, ec=BLU, fs=7.4)
        ax.add_patch(FancyArrowPatch((x+0.16, ybox+bh), (x+bw-0.16, ybox+bh), arrowstyle="-|>",
                     mutation_scale=8, lw=0.9, color=TEAL, connectionstyle="arc3,rad=-0.32", zorder=5))
        if i < 4: arrow(ax, x+bw, cy, x+bw+gap, cy)
    last_right = bx0 + 4*(bw+gap) + bw
    arrow(ax, last_right, cy, last_right+agap, cy)
    rbox(ax, last_right+agap, cy-0.65, dw, 1.3, "Dense\n\u2192 2 targets:\nGWETROOT,\nSPEI", fc=SAND, ec=ACC, fs=7.4)
    ax.text(6.3, 1.95, "receptive field", ha="center", fontsize=8.2, weight="bold", color=INK)
    ax.text(6.3, 1.42, r"$\mathrm{RF}=1+2(k-1)\sum_j d_j = 1+2(2)(31)=125\ \mathrm{days}\ \geq\ 90\text{-day window}$",
            ha="center", fontsize=9.0, color=ACC)
    fig.savefig(f"{OUT}/figure6_tcn.png"); plt.close(fig); print("fig6 tcn")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig1_framework(); fig2_studyarea(); fig3_leakage(); fig4_acf(); fig5_experiment(); fig6_tcn()
    print("ALL FIGURES REGENERATED")
