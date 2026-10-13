"""Step 6. Figures 1-6 (PNG, 600 dpi, Arial-compatible font) and TIFF copies for submission.

Fig. 1 needs the Natural Earth coastlines (public domain), downloaded once into paper_analysis_rev6/_geo/.
Fig. 2 is a schematic; Figs. 3-6 read the tables written by steps 1-4.
Output: paper_outputs_rev6/figures/Fig1_study_area.png ... Fig6_transfer.png and Fig1.tif ... Fig6.tif
"""
# ======================= Fig. 1: study area =======================
# Study area: Ten NASA POWER grid cells coloured by mean dry-season (June-September) precipitation,
# 1996-2016 (Table 1), Natural Earth coastlines, Indonesia locator inset.
import json, urllib.request, numpy as np, matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
from _common import FIG, GEO
from matplotlib.patches import Polygon, Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"], "font.size": 8})

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
fig, ax = plt.subplots(figsize=(6.85, 3.15))
land(ax, "10m", (X0 - 1, X1 + 1, Y0 - 1, Y1 + 1), 0.5)
ax.set_xlim(X0, X1); ax.set_ylim(Y0, Y1); ax.set_aspect(1 / np.cos(np.radians(9.3)))
ax.grid(color="white", lw=0.6, alpha=0.9, zorder=0.5)
sc = ax.scatter([c[1] for c in C], [c[2] for c in C], c=[c[3] for c in C], cmap=cmap, norm=norm, s=95, ec="white", lw=1.4, zorder=4)
for cid, lon, lat, _ in C:
    dx, dy, ha = LAB[cid]
    ax.text(lon + dx, lat + dy, cid, ha=ha, va="center", fontsize=8, weight="bold", color="#2A2320", zorder=5)
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
fig.savefig(FIG / "Fig1_study_area.png", dpi=600, bbox_inches="tight")


plt.close('all')

# ======================= Figs. 2-6 =======================
plt.rcParams.update(plt.rcParamsDefault)
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from _common import IN, TR, TAB, FIG
IN, TR, T, F = str(IN), str(TR), str(TAB), str(FIG)
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','Liberation Sans','DejaVu Sans'],'font.size':8,'axes.linewidth':0.6,'axes.edgecolor':'#52514e',
  'axes.labelcolor':'#0b0b0b','xtick.color':'#52514e','ytick.color':'#52514e','axes.grid':True,'grid.color':'#e6e5e1',
  'grid.linewidth':0.5,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,'savefig.dpi':600})
C={'RandomForest':'#2a78d6','TCN':'#eb6834','LSTM':'#1baf7a','Transformer':'#6250d6','Persistence':'#0b0b0b','Climatology':'#9a9893'}
MK={'RandomForest':'s','TCN':'o','LSTM':'^','Transformer':'D','Persistence':'P','Climatology':'x'}
LAB={'RandomForest':'Random Forest','TCN':'TCN','LSTM':'LSTM','Transformer':'Transformer','Persistence':'Persistence','Climatology':'Climatology'}
H=[1,7,14,30]; L=['RandomForest','TCN','LSTM','Transformer']; mm=1/25.4

# ---------- Fig 2: design schematic
fig,ax=plt.subplots(figsize=(174*mm,70*mm)); ax.axis('off'); ax.set_xlim(1995,2032); ax.set_ylim(-0.5,5.6)
def bar(y,a,b,col,txt,tc='white',fs=7.5):
    ax.add_patch(FancyBboxPatch((a,y-0.28),b-a,0.56,boxstyle='round,pad=0,rounding_size=0.15',fc=col,ec='none'))
    ax.text((a+b)/2,y,txt,ha='center',va='center',color=tc,fontsize=fs,linespacing=0.95)
ax.text(1995,5.25,'(a) Chronological protocol (each of 10 cells; leads 1, 7, 14, 30 days)',fontsize=8.5,weight='bold',ha='left')
bar(4.4,1996,2017,'#2a78d6','Development fit 1996–2016')
bar(4.4,2017,2021,'#eda100','Validation\n2017–20',tc='#0b0b0b',fs=6.5)
ax.text(2026.4,4.4,'→ fit post-processing\n   (QM, VI, THR delta)',va='center',fontsize=7.2)
bar(3.5,1996,2021,'#52514e','Final fit 1996–2020')
bar(3.5,2021,2026,'#e34948','Test 2021–25')
ax.text(2026.4,3.5,'→ apply post-processing,\n   verify once',va='center',fontsize=7.2)
for x in (1996,2001,2006,2011,2016,2021,2026): ax.text(x,2.95,str(x),ha='center',fontsize=6.5,color='#52514e')
ax.text(1995,2.45,'(b) Three verification schemes for an alert level (e.g. D1); persistence carries the issue-day state forward',fontsize=8.5,weight='bold',ha='left')
sch=[('Day-based','every target day; hits, misses and false alarms by day'),
     ('Onset','only issue days whose observed state is below the level; event = level reached at the target day'),
     ('Episode','observed and forecast drought episodes (≥7 d, gaps <10 d merged); hit = forecast onset within ±7 d')]
for i,(a,b) in enumerate(sch):
    y=1.75-0.72*i; bar(y,1996,2003.2,['#1baf7a','#6250d6','#eb6834'][i],a); ax.text(2004,y,b,va='center',fontsize=7.4)
plt.savefig(F+'/Fig2_design.png',bbox_inches='tight'); plt.close()

# ---------- Fig 3: dispersion and MSE cost (in-domain)
c=pd.read_csv(IN+'/continuous_index_space.csv')
fig,axs=plt.subplots(1,3,figsize=(174*mm,58*mm))
for k,(ix,ttl) in enumerate([('spei','SPEI-1'),('ssi','SSI')]):
    a=axs[k]
    for m in L:
        for pp,ls in (('raw','--'),('QM','-')):
            s=c[(c.model==m)&(c.postproc==pp)&(c['index']==ix)].sort_values('horizon')
            a.plot(s.horizon,s.sd_ratio,ls=ls,marker=MK[m],ms=3.5,lw=1.2,color=C[m],label=f'{LAB[m]} {"raw" if pp=="raw" else "QM"}')
    a.axhline(1,color='#0b0b0b',lw=0.8); a.set_xticks(H); a.set_xlabel('Lead time (days)')
    a.set_title(f'({"ab"[k]}) Dispersion, {ttl}',fontsize=8,loc='left'); a.set_ylim(0.2,1.3)
axs[0].set_ylabel('SD forecast / SD observed')
a=axs[2]
for m in L:
    for pp,ls in (('raw','--'),('QM','-')):
        s=c[(c.model==m)&(c.postproc==pp)&(c['index']=='spei')].sort_values('horizon')
        s=s[s.horizon>=7]; a.plot(s.horizon,s.SS_vs_persist_raw,ls=ls,marker=MK[m],ms=3.5,lw=1.2,color=C[m])
a.axhline(0,color='#0b0b0b',lw=0.8); a.set_xticks([7,14,30]); a.set_xlabel('Lead time (days)'); a.set_ylabel('MSE skill vs persistence')
a.set_title('(c) Price in squared error, SPEI-1',fontsize=8,loc='left')
from matplotlib.lines import Line2D
hd=[Line2D([],[],color=C[m],marker=MK[m],lw=1.2,ms=3.5,label=LAB[m]) for m in L]+[Line2D([],[],color='#52514e',ls='--',label='raw'),Line2D([],[],color='#52514e',ls='-',label='quantile-mapped')]
fig.legend(handles=hd,loc='lower center',ncol=6,fontsize=7,bbox_to_anchor=(0.5,-0.06)); plt.tight_layout(rect=(0,0.06,1,1))
plt.savefig(F+'/Fig3_dispersion.png',bbox_inches='tight'); plt.close()

# ---------- Fig 4: CSI day & onset, and dCSI (in-domain, D1, OR)
fig,axs=plt.subplots(2,2,figsize=(174*mm,120*mm))
for col,(f,ttl) in enumerate([('warning_day','day-based'),('warning_onset','onset')]):
    d=pd.read_csv(f'{T}/T_indomain_{f}_D1.csv')
    a=axs[0,col]; p=d[(d.model=='Persistence')&(d.postproc=='THR')].sort_values('horizon')
    a.fill_between(p.horizon,p.CSI_lo,p.CSI_hi,color='#0b0b0b',alpha=0.10,lw=0)
    a.plot(p.horizon,p.CSI,color='#0b0b0b',marker='P',ms=4,lw=1.6,label='Persistence')
    pr=d[(d.model=='Persistence')&(d.postproc=='raw')].sort_values('horizon'); a.plot(pr.horizon,pr.CSI,color='#0b0b0b',marker='P',ms=4,lw=1.2,ls='--')
    for m in L:
        for pp,ls in (('raw','--'),('THR','-')):
            s=d[(d.model==m)&(d.postproc==pp)].sort_values('horizon')
            a.plot(s.horizon,s.CSI,ls=ls,marker=MK[m],ms=3.5,lw=1.2,color=C[m])
    a.set_xticks(H); a.set_ylim(0,1 if col==0 else 0.35); a.set_ylabel('CSI'); a.set_title(f'({"ab"[col]}) CSI, {ttl} (D1, OR rule)',fontsize=8,loc='left')
    a=axs[1,col]; off={'RandomForest':-1.2,'TCN':-0.4,'LSTM':0.4,'Transformer':1.2}
    for m in L:
        s=d[(d.model==m)&(d.postproc=='THR')].sort_values('horizon')
        x=np.array([{1:1,7:7,14:14,30:30}[h] for h in s.horizon])+off[m]
        a.errorbar(x,s.dCSI_vsPersSame,yerr=[s.dCSI_vsPersSame-s.dCSI_vsPersSame_lo,s.dCSI_vsPersSame_hi-s.dCSI_vsPersSame],fmt=MK[m],ms=4,color=C[m],elinewidth=1,capsize=0)
        a.plot(x,s.dCSI_vsPersRaw,ls='none',marker=MK[m],ms=3.5,mfc='white',mec=C[m],mew=0.8)
        sig=s[s.pBH_vsPersSame<0.05]
        for xx,yy in zip(np.array([{1:1,7:7,14:14,30:30}[h] for h in sig.horizon])+off[m],sig.dCSI_vsPersSame_hi): a.text(xx,yy+0.012,'*',ha='center',fontsize=9,color='#0b0b0b')
    a.axhline(0,color='#0b0b0b',lw=0.8); a.set_xticks(H); a.set_xlabel('Lead time (days)'); a.set_ylabel('ΔCSI vs THR persistence')
    a.set_title(f'({"cd"[col]}) ΔCSI, THR models − THR persistence, {ttl}',fontsize=8,loc='left')
hd=[Line2D([],[],color='#0b0b0b',marker='P',lw=1.6,label='Persistence (THR: 95% CI shaded)')]+[Line2D([],[],color=C[m],marker=MK[m],lw=1.2,ms=3.5,label=LAB[m]) for m in L]+[Line2D([],[],color='#52514e',ls='--',label='raw'),Line2D([],[],color='#52514e',ls='-',label='threshold-shifted (THR)'),Line2D([],[],color='#52514e',marker='o',mfc='white',ls='none',label='open: vs raw persistence')]
fig.legend(handles=hd,loc='lower center',ncol=4,fontsize=7,bbox_to_anchor=(0.5,-0.04)); plt.tight_layout(rect=(0,0.06,1,1))
plt.savefig(F+'/Fig4_csi.png',bbox_inches='tight'); plt.close()

# ---------- Fig 5: episodes (hits / 63, cell-bootstrap CI) in-domain
e=pd.read_csv(T+'/T_episode_vsPersTHR.csv'); e=e[e.experiment=='indomain']
fig,axs=plt.subplots(1,2,figsize=(174*mm,62*mm))
order=['Persistence']+L
for k,h in enumerate([7,14]):
    a=axs[k]; s=e[e.horizon==h]; x=np.arange(len(order)); w=0.38
    for j,pp in enumerate(['raw','THR']):
        v=[s[(s.model==m)&(s.postproc==pp)].iloc[0] for m in order]
        a.bar(x+(j-0.5)*w,[r.fc for r in v],w-0.04,fill=False,edgecolor=[C[m] for m in order],lw=0.7,ls=(0,(2,1.5)))
        a.bar(x+(j-0.5)*w,[r.hits for r in v],w-0.04,color=[C[m] for m in order],alpha=0.45 if pp=='raw' else 1.0,
              hatch='////' if pp=='raw' else None,edgecolor='white',lw=0.5)
        for xx,r in zip(x+(j-0.5)*w,v): a.text(xx,r.hits+0.8,f'{int(r.hits)}',ha='center',fontsize=6.5,color='#0b0b0b')
        for xx,r in zip(x+(j-0.5)*w,v): a.text(xx,r.fc+0.8,f'{int(r.fc)}',ha='center',fontsize=5.5,color='#52514e')
    a.axhline(63,color='#52514e',lw=0.8,ls=':'); a.set_axisbelow(True)
    a.set_xticks(x); a.set_xticklabels([LAB[m] for m in order],rotation=20,fontsize=7); a.set_ylim(0,122)
    a.set_ylabel('Episode onsets'); a.set_title(f'({"ab"[k]}) Lead {h} days',fontsize=8,loc='left'); a.grid(axis='x',visible=False)
from matplotlib.patches import Patch
fig.legend(handles=[Patch(fc='#9a9893',alpha=0.45,hatch='////',ec='white',label='raw'),Patch(fc='#52514e',label='threshold-shifted (THR)'),Patch(fill=False,ec='#52514e',ls=(0,(2,1.5)),label='forecast episode onsets (hits + false alarms)'),Line2D([],[],color='#52514e',lw=0.8,ls=':',label='63 observed onsets')],loc='lower center',ncol=4,fontsize=7,bbox_to_anchor=(0.5,-0.03))
plt.tight_layout(rect=(0,0.06,1,1)); plt.savefig(F+'/Fig5_episodes.png',bbox_inches='tight'); plt.close()

# ---------- Fig 6: in-domain vs transfer (TCN, RF; THR) dCSI vs persistence + episodes
fig,axs=plt.subplots(1,3,figsize=(174*mm,58*mm))
for k,(f,ttl) in enumerate([('warning_day','Day-based ΔCSI'),('warning_onset','Onset ΔCSI')]):
    a=axs[k]
    for exp,ls,mk_off in (('indomain','-',-0.6),('transfer',':',0.6)):
        d=pd.read_csv(f'{T}/T_{exp}_{f}_D1.csv')
        for m in ['RandomForest','TCN']:
            s=d[(d.model==m)&(d.postproc=='THR')].sort_values('horizon'); x=s.horizon+mk_off+(0.25 if m=='TCN' else -0.25)
            a.errorbar(x,s.dCSI_vsPersSame,yerr=[s.dCSI_vsPersSame-s.dCSI_vsPersSame_lo,s.dCSI_vsPersSame_hi-s.dCSI_vsPersSame],fmt=MK[m],ms=4,
                       color=C[m],mfc=C[m] if exp=='indomain' else 'white',elinewidth=1,ls='none')
    a.axhline(0,color='#0b0b0b',lw=0.8); a.set_xticks(H); a.set_xlabel('Lead time (days)'); a.set_title(f'({"ab"[k]}) {ttl}, THR, D1',fontsize=8,loc='left')
axs[0].set_ylabel('ΔCSI vs THR persistence')
a=axs[2]; x=np.arange(4); w=0.2
for j,(exp,m) in enumerate([('indomain','RandomForest'),('transfer','RandomForest'),('indomain','TCN'),('transfer','TCN')]):
    s=e0=pd.read_csv(T+'/T_episode_cellboot.csv'); s=s[(s.experiment==exp)&(s.model==m)&(s.postproc=='THR')].sort_values('horizon')
    a.bar(x+(j-1.5)*w,s.hits,w-0.03,color=C[m],alpha=1 if exp=='indomain' else 0.4,edgecolor=C[m],lw=0.8)
p=e0[(e0.experiment=='indomain')&(e0.model=='Persistence')&(e0.postproc=='THR')].sort_values('horizon')
a.plot(x,p.hits,color='#0b0b0b',marker='P',lw=1.4,label='Persistence (THR)')
a.set_xticks(x); a.set_xticklabels(H); a.set_xlabel('Lead time (days)'); a.set_ylabel('Onsets hit (of 63)'); a.set_title('(c) Episode onsets hit',fontsize=8,loc='left'); a.grid(axis='x',visible=False)
hd=[Line2D([],[],color=C['RandomForest'],marker='s',ls='none',label='Random Forest, in-domain'),Line2D([],[],color=C['RandomForest'],marker='s',mfc='white',ls='none',label='Random Forest, transfer'),
    Line2D([],[],color=C['TCN'],marker='o',ls='none',label='TCN, in-domain'),Line2D([],[],color=C['TCN'],marker='o',mfc='white',ls='none',label='TCN, transfer'),Line2D([],[],color='#0b0b0b',marker='P',label='Persistence (THR)')]
fig.legend(handles=hd,loc='lower center',ncol=5,fontsize=7,bbox_to_anchor=(0.5,-0.06)); plt.tight_layout(rect=(0,0.07,1,1))
plt.savefig(F+'/Fig6_transfer.png',bbox_inches='tight'); plt.close()


# ======================= TIFF copies (Springer: Fig1.tif ... Fig6.tif) =======================
from PIL import Image
import glob as _glob, os as _os
for _f in sorted(_glob.glob(str(FIG / "Fig*_*.png"))):
    _n = _os.path.basename(_f).split("_")[0]
    Image.open(_f).convert("RGB").save(FIG / f"{_n}.tif", compression="tiff_lzw", dpi=(600, 600))
print("wrote", FIG)
