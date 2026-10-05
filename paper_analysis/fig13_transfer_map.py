"""Figure 13: transfer skill retention map and in-domain vs transfer skill per cell (14-day lead, TCN). Input: results_mc/skill_map.csv."""
from _paths import RMC, OUT
import pandas as pd, numpy as np, matplotlib as mpl
mpl.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from _basemap import draw_land
plt.rcParams.update({'font.size':10,'font.family':'DejaVu Sans','axes.linewidth':0.8,'savefig.dpi':300})
sm=pd.read_csv(RMC / 'skill_map.csv')
sm['ret']=sm.SS_transfer/sm.SS_indomain*100
lbl={'lombok':'Lombok','sumbawaB':'W Sumbawa','sumbawaC':'C Sumbawa','bima':'Bima','manggarai':'Manggarai',
     'ende':'Ende','floresT':'E Flores','sumba':'Sumba','kupang':'Kupang','soe':'Soe'}
sm['short']=sm.cell_id.map(lbl)
fig=plt.figure(figsize=(9.0,8.6))
gs=fig.add_gridspec(2,2,height_ratios=[1.0,1.05],width_ratios=[1,0.028],hspace=0.30,wspace=0.03,
                    left=0.115,right=0.93,top=0.95,bottom=0.07)
ax=fig.add_subplot(gs[0,0]); cax=fig.add_subplot(gs[0,1])
draw_land(ax,lw=0.6)
norm=TwoSlopeNorm(vmin=0,vcenter=100,vmax=120); cmap=plt.cm.RdYlBu
sc=ax.scatter(sm.lon,sm.lat,s=90+sm.SS_transfer*520,c=sm.ret,cmap=cmap,norm=norm,edgecolors='#333',linewidths=0.9,zorder=3)
mg=sm.loc[sm.ret.idxmin()]
ax.scatter([mg.lon],[mg.lat],s=430,facecolors='none',edgecolors='#B2182B',linewidths=2.0,linestyle=(0,(3,2)),zorder=4)
ax.annotate(f'outlier: {mg.ret:.0f}% retention',xy=(mg.lon+0.12,mg.lat-0.25),xytext=(121.55,-9.62),fontsize=9,color='#B2182B',
            weight='bold',ha='left',va='center',arrowprops=dict(arrowstyle='-',color='#B2182B',lw=1.0),zorder=5)
offs={'lombok':(-0.05,0.33),'sumbawaB':(0.05,-0.36),'sumbawaC':(0.05,0.33),'bima':(0.05,0.33),'manggarai':(0,0.36),
      'ende':(0,0.33),'floresT':(0,0.35),'sumba':(0,-0.40),'kupang':(-0.1,-0.42),'soe':(0.1,0.34)}
for _,r in sm.iterrows():
    dx,dy=offs[r.cell_id]
    ax.text(r.lon+dx,r.lat+dy,r['short'],fontsize=9,ha='center',va='bottom' if dy>0 else 'top',color='#2A2320',zorder=5,
            bbox=dict(boxstyle='round,pad=0.12',fc='white',ec='none',alpha=0.65))
ax.set_xlim(115.5,125.2); ax.set_ylim(-11.55,-7.45); ax.set_aspect(1.01)
ax.set_xlabel('Longitude (°E)'); ax.set_ylabel('Latitude (°N)')
ax.grid(alpha=0.5,lw=0.4,color='white',zorder=2)
ax.set_title('(a)  Transfer skill retention across the dry-season gradient',fontsize=11,weight='bold',loc='left',pad=8)
ax.text(117.35,-11.0,'NTB',fontsize=9,color='#6E635C',ha='center',style='italic')
ax.text(122.2,-11.0,'NTT',fontsize=9,color='#6E635C',ha='center',style='italic')
ax.plot([116.0,118.8],[-11.17,-11.17],color='#9c928a',lw=1.2); ax.plot([119.8,124.6],[-11.17,-11.17],color='#9c928a',lw=1.2)
h=[Line2D([0],[0],marker='o',color='w',markerfacecolor='#d9d9d9',markeredgecolor='#333',
          markersize=np.sqrt(90+v*520),markeredgewidth=0.9,label=f'{v:.1f}') for v in [0.1,0.3,0.5]]
ax.legend(handles=h,title='SS transfer (marker size)',loc='center',bbox_to_anchor=(0.215,0.335),fontsize=8,title_fontsize=8,ncol=3,framealpha=0.9,
          handletextpad=0.8,columnspacing=1.6,borderpad=0.9,labelspacing=1.2)
cb=fig.colorbar(sc,cax=cax,ticks=[0,25,50,75,100,120]); cb.set_label('Retention of in-domain skill (%)',fontsize=9)
cb.ax.axhline(100,color='#333',lw=1.0)
ax2=fig.add_subplot(gs[1,:])
d=sm.sort_values('ret').reset_index(drop=True)
for i,r in d.iterrows():
    col='#B2182B' if r.SS_transfer<r.SS_indomain else '#2166AC'
    ax2.plot([r.SS_indomain,r.SS_transfer],[i,i],color=col,lw=2.2,alpha=0.75,zorder=2,solid_capstyle='round')
    ax2.scatter(r.SS_indomain,i,s=50,facecolor='white',edgecolor='#555',lw=1.2,zorder=3)
    ax2.scatter(r.SS_transfer,i,s=56,color=col,edgecolor='#333',lw=0.8,zorder=4)
    ax2.text(0.655,i,f'{r.ret:.0f}%',fontsize=9,va='center',color=col,weight='bold' if r.ret<50 else 'normal')
ax2.set_yticks(np.arange(len(d))); ax2.set_yticklabels(d['short'],fontsize=9)
ax2.set_xlim(min(0.0, float(d.SS_transfer.min()) - 0.02), 0.70); ax2.set_ylim(-0.7,len(d)-0.3)
ax2.set_xlabel('Skill score relative to persistence (root-zone soil moisture)')
ax2.grid(axis='x',alpha=0.25,lw=0.5); ax2.set_axisbelow(True)
for s in ['top','right']: ax2.spines[s].set_visible(False)
ax2.set_title('(b)  In-domain and transfer skill by cell',fontsize=11,weight='bold',loc='left',pad=8)
leg=[Line2D([0],[0],marker='o',color='w',markerfacecolor='white',markeredgecolor='#555',markersize=7,label='in-domain'),
     Line2D([0],[0],marker='o',color='w',markerfacecolor='#B2182B',markeredgecolor='#333',markersize=7,label='transfer (loss)'),
     Line2D([0],[0],marker='o',color='w',markerfacecolor='#2166AC',markeredgecolor='#333',markersize=7,label='transfer (gain)')]
ax2.legend(handles=leg,loc='upper left',fontsize=8.5,framealpha=0.95)
fig.savefig(OUT / 'fig13_transfer_map.png',dpi=300,facecolor='white')
fig.savefig(OUT / 'fig13_transfer_map.pdf',facecolor='white')
print('ok')
