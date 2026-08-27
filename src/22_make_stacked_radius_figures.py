from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import fitz

import argparse
_parser=argparse.ArgumentParser()
_parser.add_argument('--project-root', default='.')
_args=_parser.parse_args()
ROOT=Path(_args.project_root).resolve()
FIG=ROOT/'figures'

# ---------- Figure 10: radius-error curves, stacked ----------
df=pd.read_csv(ROOT/'data/error_summary_by_config_target.csv')
order=['parameter_only','local','R0p05','R0p10','R0p20','R0p35','R0p50','R0p75','R1','R1p50','R2','R3','full']
labels=['Param.','Local','0.05','0.10','0.20','0.35','0.50','0.75','1.0','1.5','2.0','3.0','Full']
targets=[('Cp',r'$C_p$'),('Cq',r'$C_q$'),('tau_abs',r'$|\tau|$')]
plt.rcParams.update({
    'font.size':14,
    'axes.labelsize':15,
    'axes.titlesize':16,
    'xtick.labelsize':12,
    'ytick.labelsize':13,
    'legend.fontsize':12,
    'pdf.fonttype':42,
    'ps.fonttype':42,
    'axes.linewidth':1.2,
})
fig,axes=plt.subplots(3,1,figsize=(9.0,10.0),constrained_layout=False)
fig.subplots_adjust(left=0.11,right=0.97,bottom=0.075,top=0.90,hspace=0.50)
x=np.arange(len(order))
for k,(ax,(target,tlabel)) in enumerate(zip(axes,targets)):
    sub=df[df.target==target].set_index('config')
    y=[100*float(sub.loc[c,'relL2']) for c in order]
    ax.plot(x,y,marker='o',linewidth=2.3,markersize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels,rotation=0,ha='center')
    ax.set_ylabel('Mean LOOCV error (%)')
    ax.set_xlabel(r'Model input / observed radius $R/h_s$')
    ax.set_title(f'({chr(97+k)}) {tlabel}',loc='left',pad=6)
    ax.grid(True,axis='y',alpha=0.28)
    ax.margins(x=0.02)
fig.suptitle('Bulk-to-wall surface-load error versus observed bulk-field radius',fontsize=18,y=0.975)
fig.savefig(FIG/'fig10_bulk_to_wall_error.pdf',bbox_inches='tight')
fig.savefig(FIG/'fig10_bulk_to_wall_error.png',dpi=400,bbox_inches='tight')
plt.close(fig)

# ---------- Figure 12: recover median and percentile bands from vector PDF ----------
src=ROOT/'data'/'fig13_closed_loop_error_curves_original_side_by_side.pdf'
doc=fitz.open(src)
page=doc[0]
draws=page.get_drawings()
blue=(0.12156862765550613,0.46666666865348816,0.7058823704719543)
red=(0.8392156958580017,0.15294118225574493,0.1568627506494522)
# exact line/fill drawing indices grouped by panel (left, middle, right)
panel_cfg=[
    {'line_blue':None,'line_red':None,'fill_blue':None,'fill_red':None,'y0':414.3979797363281,'dy':10.304864501953125,'label':r'$C_p$'},
    {'line_blue':None,'line_red':None,'fill_blue':None,'fill_red':None,'y0':404.781494140625,'dy':4.0499542236328125,'label':r'$C_q$'},
    {'line_blue':None,'line_red':None,'fill_blue':None,'fill_red':None,'y0':410.8476257324219,'dy':3.628887939453125,'label':r'$|\tau|$'},
]
# classify by x-center and type/color
for d in draws:
    rect=d['rect']; xc=(rect.x0+rect.x1)/2
    if xc<435: pi=0
    elif xc<870: pi=1
    elif xc>880: pi=2
    else: continue
    color=d.get('color'); fill=d.get('fill'); width=d.get('width') or 0
    if d.get('type')=='s' and abs(width-2.4)<0.05:
        if color==blue and len(d['items'])>=12: panel_cfg[pi]['line_blue']=d
        if color==red and len(d['items'])>=12: panel_cfg[pi]['line_red']=d
    if d.get('type')=='fs' and len(d['items'])==26 and d.get('fill_opacity',1)<0.5:
        if fill==blue: panel_cfg[pi]['fill_blue']=d
        if fill==red: panel_cfg[pi]['fill_red']=d

def line_y(d):
    items=d['items']
    ys=[items[0][1].y]
    for item in items:
        ys.append(item[2].y)
    # first start + all ends gives 13, but final count can have duplicate pattern; take first 13
    return np.array(ys[:13],float)

def band_y(d):
    items=d['items']
    # lower-data-value boundary: item0 end + item1..12 ends (PDF coordinates larger)
    low_pdf=[items[0][2].y]+[items[i][2].y for i in range(1,13)]
    # upper-data-value boundary: item13 end at x12, then item14..25 ends back to x0
    up_rev=[items[13][2].y]+[items[i][2].y for i in range(14,26)]
    up_pdf=list(reversed(up_rev))
    return np.array(low_pdf,float),np.array(up_pdf,float)

plt.rcParams.update({
    'font.size':14,
    'axes.labelsize':15,
    'axes.titlesize':16,
    'xtick.labelsize':12,
    'ytick.labelsize':13,
    'legend.fontsize':13,
    'pdf.fonttype':42,
    'ps.fonttype':42,
    'axes.linewidth':1.2,
})
fig,axes=plt.subplots(3,1,figsize=(9.0,10.3),constrained_layout=False)
fig.subplots_adjust(left=0.11,right=0.97,bottom=0.075,top=0.88,hspace=0.52)
x=np.arange(13)
for k,(ax,cfg) in enumerate(zip(axes,panel_cfg)):
    for dline,dfill,label in [(cfg['line_blue'],cfg['fill_blue'],'Raw DSMC field'),(cfg['line_red'],cfg['fill_red'],'Reconstructed surrogate field')]:
        med=(cfg['y0']-line_y(dline))/cfg['dy']
        low_pdf,up_pdf=band_y(dfill)
        lower=(cfg['y0']-low_pdf)/cfg['dy']
        upper=(cfg['y0']-up_pdf)/cfg['dy']
        # Matplotlib uses same default sequence as original: blue then red.
        line=ax.plot(x,med,marker='o',linewidth=2.3,markersize=6.0,label=label)[0]
        ax.fill_between(x,lower,upper,color=line.get_color(),alpha=0.18,linewidth=0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels,rotation=0,ha='center')
    ax.set_ylabel('Median LOOCV error (%)')
    ax.set_xlabel(r'Model input / observed radius $R/h_s$')
    ax.set_title(f'({chr(97+k)}) {cfg["label"]}',loc='left',pad=6)
    ax.grid(True,axis='y',alpha=0.28)
    ax.margins(x=0.02)
handles,leglabels=axes[0].get_legend_handles_labels()
fig.legend(handles,leglabels,loc='upper center',bbox_to_anchor=(0.5,0.935),ncol=2,frameon=False)
fig.suptitle('Closed-loop preservation of bulk-to-wall error curves',fontsize=18,y=0.985)
fig.savefig(FIG/'fig13_closed_loop_error_curves.pdf',bbox_inches='tight')
fig.savefig(FIG/'fig13_closed_loop_error_curves.png',dpi=400,bbox_inches='tight')
plt.close(fig)

print('wrote stacked figures')
