"""Figures 4 (autocorrelation of the targets, 1996-2016) and 6 (TCN architecture). Drawing code in code/make_all_figures_pro.py."""
import os
from _paths import OUT, DATA
os.environ.setdefault("FIG_OUT", str(OUT)); os.environ.setdefault("BASE_CELL_CSV", str(DATA / "cell_sumbawaC.csv"))
import make_all_figures_pro as F
F.fig4_acf(); F.fig6_tcn()
for a, b in (("figure4_acf.png", "fig04_acf.png"), ("figure6_tcn.png", "fig06_tcn.png")):
    if (OUT / a).exists(): (OUT / a).replace(OUT / b)
