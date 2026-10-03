"""Graphical abstract (1536 x 614 px, EMS). Inputs: results/, results_mc/skill_map.csv, paper_outputs/dissociation_matrix.csv."""
from _paths import RES, RMC, OUT
import pandas as pd, numpy as np, matplotlib as mpl
mpl.use('Agg'); import matplotlib.pyplot as plt
from _basemap import draw_land
plt.rcParams.update({'font.family':'DejaVu Sans','axes.linewidth':0.6})
sm=pd.read_csv(RMC/'skill_map.csv')
mx=pd.read_csv(OUT/'dissociation_matrix.csv')
MAROON='#7A1523'; GOLD='#C77A18'; BLUE='#2166AC'; RED='#B2182B'; GREY='#6E635C'
fig=plt.figure(figsize=(5.12,2.048),dpi=300); fig.patch.set_facecolor('white')
fig.text(0.5,0.955,'Forecast skill is not warning skill',ha='center',va='center',fontsize=9.2,weight='bold',color=MAROON)
fig.text(0.5,0.877,'Drought forecasting across the Nusa Tenggara dry-season gradient, Indonesia',ha='center',va='center',fontsize=4.8,color=GREY)
gs=fig.add_gridspec(1,3,width_ratios=[1.75,0.92,1.12],left=0.012,right=0.985,top=0.74,bottom=0.235,wspace=0.36)

# (1) MAP
ax=fig.add_subplot(gs[0,0])
draw_land(ax,lw=0.35)
ax.scatter(sm.lon,sm.lat,s=11,c=MAROON,edgecolors='white',linewidths=0.35,zorder=3)
ax.set_xlim(115.75,125.0); ax.set_ylim(-11.3,-7.6); ax.set_aspect(1.01)
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values(): s.set_color('#bbbbbb'); s.set_linewidth(0.5)
ax.set_title('10 grid cells · NASA POWER 1996–2025',fontsize=4.7,color='#2A2320',pad=2.5)
for nm,x,y in [('Lombok',116.3,-9.35),('Sumbawa',117.9,-8.05),('Flores',121.3,-8.1),('Sumba',119.9,-10.05),('Timor',124.3,-10.7)]:
    ax.text(x,y,nm,fontsize=3.6,color=GREY,ha='center',style='italic',zorder=4)
ax.annotate('',xy=(124.7,-11.05),xytext=(116.0,-11.05),arrowprops=dict(arrowstyle='->',color=GOLD,lw=0.8),zorder=4)
ax.text(120.3,-10.93,'drier eastward',ha='center',va='bottom',fontsize=3.9,color=GOLD,weight='bold',zorder=4)

# (2) BARS
ax2=fig.add_subplot(gs[0,1])
lab=['Persist.','TCN','LSTM','RF']; x=np.arange(4); w=0.38
raw=pd.read_csv(RES/'metrics_raw.csv'); dv=pd.read_csv(RES/'dss_validation.csv')
MODS=['Persistence','TCN','LSTM','RandomForest']
ss=raw[(raw.target=='GWETROOT')&(raw.horizon==30)].groupby('model').SS_vs_persist.mean()
cont=[0.0 if m=='Persistence' else ss[m] for m in MODS]
cs=dv[(dv.horizon==30)&(dv.logic=='or')&(dv.level=='D1')].set_index('model').CSI
warn=[cs[m] for m in MODS]
ax2.bar(x-w/2,cont,w,color=BLUE,label='accuracy (SS)'); ax2.bar(x+w/2,warn,w,color=RED,label='warning (CSI)')
ax2.set_xticks(x); ax2.set_xticklabels(lab,fontsize=4.2)
ax2.tick_params(axis='y',labelsize=4.0,length=1.5,pad=1)
ax2.set_ylim(0,0.88); ax2.set_yticks([0,0.3,0.6]); ax2.axhline(0,color='#888',lw=0.5)
for s in ['top','right']: ax2.spines[s].set_visible(False)
ax2.set_title('At 30-day lead',fontsize=5.0,color='#2A2320',pad=2.5)
ax2.legend(fontsize=3.8,frameon=False,loc='upper center',bbox_to_anchor=(0.52,1.03),handlelength=0.8,handletextpad=0.3,labelspacing=0.15)
ax2.text(3.0,0.68,'best\naccuracy',ha='center',va='bottom',fontsize=3.8,color=BLUE,linespacing=1.1)
ax2.text(0.10,0.30,'best\nwarning',ha='center',va='bottom',fontsize=3.8,color=RED,linespacing=1.1)

# (3) RHO
ax3=fig.add_subplot(gs[0,2])
ax3.axhspan(-0.5,0,color='#FBECEA',zorder=0); ax3.axhline(0,color='#888',lw=0.6,ls='--')
cols=plt.cm.YlOrRd(np.linspace(0.42,0.92,4))
for k,lv in enumerate(['D0','D1','D2','D3']):
    dd=mx[(mx.source=='multicell')&(mx.target=='GWETROOT')&(mx.level==lv)&(mx.horizon!='pooled')]
    dd=dd.assign(h=dd.horizon.astype(int)).sort_values('h')
    ax3.plot(dd.h,dd.rho,'-o',color=cols[k],lw=1.0,ms=2.0,label=lv)
ax3.set_xticks([1,7,14,30]); ax3.set_xticklabels([1,7,14,30],fontsize=4.0)
ax3.tick_params(axis='y',labelsize=4.0,length=1.5,pad=1)
ax3.set_ylim(-0.48,1.05); ax3.set_yticks([-0.3,0,0.4,0.8])
ax3.set_xlabel('lead time (days)',fontsize=4.4,labelpad=1.2); ax3.set_ylabel('rank agreement  ρ',fontsize=4.4,labelpad=1.5)
for s in ['top','right']: ax3.spines[s].set_visible(False)
ax3.set_title('Agreement falls with lead & severity',fontsize=5.0,color='#2A2320',pad=2.5)
ax3.legend(fontsize=3.7,frameon=False,loc='upper right',ncol=2,handlelength=0.7,handletextpad=0.25,columnspacing=0.6,labelspacing=0.15)
ax3.text(29.5,-0.41,'dissociation',ha='right',fontsize=4.0,color=RED,weight='bold')

fig.text(0.5,0.055,'Models chosen for low continuous error gave poorer warnings than persistence',ha='center',va='center',fontsize=5.2,weight='bold',color=MAROON)
fig.savefig(OUT/'graphical_abstract.png',dpi=300,facecolor='white')
fig.savefig(OUT/'graphical_abstract.pdf',facecolor='white')
print('ok', [round(x,2) for x in cont], [round(x,2) for x in warn])
