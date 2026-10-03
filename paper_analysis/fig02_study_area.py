"""Figure 2: study area. Ten NASA POWER grid cells coloured by mean dry-season (June-September) precipitation,
1996-2016 (values as in Table 1), on Natural Earth coastlines (public domain), with an Indonesia locator inset."""
import json, urllib.request, numpy as np, matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Polygon, Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from _paths import OUT
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9})
GEO = Path(__file__).resolve().parent / "_geo"
URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_{r}_land.geojson"
def rings(res, bbox):
    f = GEO / f"ne_{res}_land.geojson"
    if not f.exists(): GEO.mkdir(exist_ok=True); urllib.request.urlretrieve(URL.format(r=res), f)
    out = []
    for ft in json.load(open(f))["features"]:
        g = ft["geometry"]; polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for p in polys:
            xs = [q[0] for q in p[0]]; ys = [q[1] for q in p[0]]
            if max(xs) >= bbox[0] and min(xs) <= bbox[1] and max(ys) >= bbox[2] and min(ys) <= bbox[3]: out.append(p[0])
    return out
def land(ax, res, bbox, lw):
    ax.add_collection(PatchCollection([Polygon(r, closed=True) for r in rings(res, bbox)], fc="#ECE7E1", ec="#6f665e", lw=lw, zorder=1))
    ax.set_facecolor("#EAF2F7")
# Table 1: id, lon, lat, dry-season P (JJAS, mm, 1996-2016)
C = [("C01",116.3,-8.6,137),("C02",116.9,-8.8,111),("C03",117.4,-8.5,72),("C04",118.5,-8.5,71),("C05",120.4,-8.6,150),
     ("C06",121.6,-8.7,83),("C07",122.9,-8.4,64),("C08",120.2,-9.6,84),("C09",123.6,-10.1,37),("C10",124.2,-9.9,34)]
LAB = {"C01":(0,.33,"center"),"C02":(0,-.36,"center"),"C03":(0,.33,"center"),"C04":(.42,.12,"left"),"C05":(0,.33,"center"),
       "C06":(0,-.36,"center"),"C07":(0,.33,"center"),"C08":(.4,0,"left"),"C09":(-.35,-.25,"right"),"C10":(.35,.15,"left")}
cmap = LinearSegmentedColormap.from_list("dry", ["#8C4A1E", "#C98A45", "#E6C98A", "#9CC3A8", "#2E8B7A"]); norm = Normalize(30, 155)
X0, X1, Y0, Y1 = 115.6, 125.3, -11.3, -7.6
fig, ax = plt.subplots(figsize=(9.2, 4.1))
land(ax, "10m", (X0 - 1, X1 + 1, Y0 - 1, Y1 + 1), 0.5)
ax.set_xlim(X0, X1); ax.set_ylim(Y0, Y1); ax.set_aspect(1 / np.cos(np.radians(9.3)))
ax.grid(color="white", lw=0.6, alpha=0.9, zorder=0.5)
sc = ax.scatter([c[1] for c in C], [c[2] for c in C], c=[c[3] for c in C], cmap=cmap, norm=norm, s=150, ec="white", lw=1.4, zorder=4)
for cid, lon, lat, _ in C:
    dx, dy, ha = LAB[cid]
    ax.text(lon + dx, lat + dy, cid, ha=ha, va="center", fontsize=8.5, weight="bold", color="#2A2320", zorder=5)
for nm, x, y in [("Lombok",116.3,-8.05),("Sumbawa",117.9,-8.05),("Flores",121.3,-8.05),("Sumba",119.75,-10.05),("Timor",124.55,-10.55)]:
    ax.text(x, y, nm, ha="center", fontsize=9, style="italic", color="#5a524b", zorder=5)
xt = np.arange(116.5, 125.1, 1.5); yt = np.arange(-11, -7.9, 0.5)
ax.set_xticks(xt); ax.set_xticklabels([f"{v:g}\u00b0E" for v in xt]); ax.set_yticks(yt); ax.set_yticklabels([f"{-v:g}\u00b0S" for v in yt])
ax.tick_params(labelsize=8.5)
sx, sy, L = 121.2, -10.95, 100 / (111.32 * np.cos(np.radians(10.9)))           # 100 km scale bar
ax.plot([sx, sx + L], [sy, sy], color="#222833", lw=2.5, solid_capstyle="butt", zorder=5)
for x in (sx, sx + L): ax.plot([x, x], [sy - .05, sy + .05], color="#222833", lw=1.2, zorder=5)
ax.text(sx + L / 2, sy + .12, "100 km", ha="center", fontsize=8.5, zorder=5)
ax.annotate("", xy=(124.95, -7.98), xytext=(124.95, -8.45), arrowprops=dict(arrowstyle="-|>", color="#222833", lw=1.4), zorder=5)
ax.text(124.95, -7.82, "N", ha="center", va="center", fontsize=10, weight="bold", zorder=5)
ins = ax.inset_axes([0.012, 0.025, 0.26, 0.30]); B = (94, 142, -11.5, 6.8)
land(ins, "50m", B, 0.2); ins.set_xlim(B[0], B[1]); ins.set_ylim(B[2], B[3]); ins.set_aspect("equal")
ins.add_patch(Rectangle((X0, Y0), X1 - X0, Y1 - Y0, fill=False, ec="#B2182B", lw=1.2))
ins.set_xticks([]); ins.set_yticks([]); ins.text(0.02, 1.03, "Indonesia", transform=ins.transAxes, ha="left", va="bottom", fontsize=7.5, style="italic", color="#5a524b")
for s in ins.spines.values(): s.set_edgecolor("#444"); s.set_linewidth(0.8)
cax = ax.inset_axes([1.015, 0, 0.022, 1]); cb = fig.colorbar(sc, cax=cax)
cb.set_label("Dry-season precipitation\n(June\u2013September, mm)", fontsize=8.5); cb.ax.tick_params(labelsize=8)
fig.savefig(OUT / "fig02_study_area.png", dpi=300, bbox_inches="tight"); fig.savefig(OUT / "fig02_study_area.pdf", bbox_inches="tight")
print("ok")
