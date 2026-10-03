"""Natural Earth 1:10m land polygons for the Nusa Tenggara maps (downloaded once, public domain)."""
import json, urllib.request
from pathlib import Path
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_land.geojson"
GEO = Path(__file__).resolve().parent / "_geo" / "ne_10m_land.geojson"
def _rings(bbox=(114.5, 126.5, -12.5, -6.5)):
    if not GEO.exists():
        GEO.parent.mkdir(exist_ok=True); urllib.request.urlretrieve(URL, GEO)
    out = []
    for f in json.load(open(GEO))["features"]:
        g = f["geometry"]; polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for poly in polys:
            xs = [q[0] for q in poly[0]]; ys = [q[1] for q in poly[0]]
            if max(xs) < bbox[0] or min(xs) > bbox[1] or max(ys) < bbox[2] or min(ys) > bbox[3]: continue
            out.append(poly[0])
    return out
def draw_land(ax, fc="#ECE7E1", ec="#7d746c", lw=0.5):
    ax.add_collection(PatchCollection([Polygon(r, closed=True) for r in _rings()], facecolor=fc, edgecolor=ec, linewidth=lw, zorder=1))
    ax.set_facecolor("#EAF2F7")
