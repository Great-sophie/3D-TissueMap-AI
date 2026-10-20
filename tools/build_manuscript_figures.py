#!/usr/bin/env python3
"""Build five manuscript-candidate figures from FROZEN 3D-TissueMap AI experiment CSVs.

No refitting, retuning, or invented values. The figures are working drafts that require
image inspection, metadata checking and author approval before journal submission.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import re
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patches

BLUE = '#176B87'
TEAL = '#25998B'
ORANGE = '#D68A39'
RED = '#BD5B54'
PURPLE = '#7868A6'
DARK = '#203244'
GRAY = '#687984'
PALE = '#ECF2F4'
C = {'nearest1':GRAY, 'nearest2':BLUE, 'all_others':ORANGE,
     'variation':GRAY, 'signal_only':TEAL, 'full':BLUE,
     'geometry':PURPLE, 'full_no_absolute_intensity':ORANGE,
     'signal_no_absolute_intensity':RED}
M = {'HER2_3B5':'HER2 3B5', 'HER2_29D8':'HER2 29D8',
     'Gd156':'Gd156 (c-erbB-?)','Cd112':'Cd112 (panCK)',
     'Eu151':'Eu151 (CD31)','Yb174':'Yb174 (E-cadherin)','Er168':'Er168 (LYVE-1)'}

INPUTS = {
 'lag':'runs/secondher2_z_lag_profile_v0_1/z_lag_error_summary.csv',
 'source_sampling':'runs/secondher2_core_study_v0_1/core_comparison_summary.csv',
 'target_sampling':'runs/lvilymph_external_sampling_v0_1/external_sampling_summary.csv',
 'target_sampling_ci':'runs/lvilymph_external_sampling_v0_1/paired_spatial_block_intervals.csv',
 'source_risk':'runs/lvilymph_portable_reliability_ablation_v0_1/source_ablation_spatial_oof.csv',
 'target_risk':'runs/lvilymph_portable_reliability_ablation_v0_1/external_ablation_summary.csv',
 'target_risk_ci':'runs/lvilymph_portable_reliability_ablation_v0_1/external_paired_spatial_block_intervals.csv',
 'domain_shift':'runs/lvilymph_portable_reliability_transfer_v0_1/feature_domain_shift.csv',
}

def setup_style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
      'axes.spines.top':False,'axes.spines.right':False,
      'axes.titlesize':11,'axes.titleweight':'semibold','axes.labelsize':9.5,
      'legend.fontsize':8.5,'figure.facecolor':'white','axes.facecolor':'white',
      'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white',
      'xtick.labelsize':9,'ytick.labelsize':9,'axes.edgecolor':'#9DADB7',
      'grid.alpha':0.16})


def panel(ax, name):
    ax.text(-.07,1.065,name,transform=ax.transAxes,ha='left',va='top',weight='bold',
            fontsize=16,color=DARK,clip_on=False)


def save(fig,out,name):
    fig.savefig(out / f'{name}.pdf',bbox_inches='tight',pad_inches=.12)
    fig.savefig(out / f'{name}.png',dpi=330,bbox_inches='tight',pad_inches=.12)
    plt.close(fig)


def load_checked(root):
    errs=[]; datasets={}
    for label,rel in INPUTS.items():
        path = root/rel
        if not path.is_file(): errs.append(str(path))
        else: datasets[label]=pd.read_csv(path)
    if errs:
        raise FileNotFoundError('Required frozen CSVs are missing (no plots generated):\n  '+'\n  '.join(errs))
    required={
      'lag':{'group','marker','z_lag','physical_lag_um','n_objects',
             'median_single_plane_error','median_paired_extra_error_vs_lag1'},
      'source_sampling':{'marker','method','n_objects','median_abs_log_error',
                         'mean_paired_improvement_vs_nearest2','bootstrap_lower_025','bootstrap_upper_975'},
      'target_sampling':{'isotope','n_objects','nearest1_median_abs_log_error',
        'nearest2_median_abs_log_error','all_others_median_abs_log_error'},
      'target_sampling_ci':{'isotope','method_a','method_b','mean_error_improvement_a_vs_b',
                            'boot_p025','boot_p975'},
      'source_risk':{'source_model','method','source_spatial_oof_auroc'},
      'target_risk':{'source_model','target_marker','method',
        'external_auroc_target_q75','external_auprc_target_q75','selective_risk_reduction_fraction'},
      'target_risk_ci':{'source_model','target_marker','contrast','delta_auroc_target_q75','lower_025','upper_975'},
      'domain_shift':{'source_model','target_marker','feature','target_fraction_outside_source_95pct_range'}
    }
    for key,fields in required.items():
        absent=fields-set(datasets[key].columns)
        if absent: raise ValueError(f'{INPUTS[key]} missing required fields: {sorted(absent)}')
    return datasets


def representative_image_secondher2(root, z=45):
    path=root/'data/inspect/SecondHer2BreastCancerModel/SecondHer2BreastCancerModel/aligned_SecondHer2BreastCancer_model_3D_image_stack.ome.tiff'
    mask=root/'data/inspect/SecondHer2BreastCancerModel/SecondHer2BreastCancerModel/measured_mask_final_segmentation_hwatershed_bg500_90%.tif'
    if not path.is_file(): return None,'Source OME-TIFF not found'
    try:
        import tifffile
        with tifffile.TiffFile(path) as tf:
            ax=tf.series[0].axes; shape=tf.series[0].shape
            if ax!='ZCYX' or len(shape)!=4 or shape[0]<46 or shape[1]<23:
                return None,f'Source OME dimension mismatch: {ax}/{shape}'
            nuc=tf.pages[z*shape[1]+11].asarray()
            prot=tf.pages[z*shape[1]+22].asarray()
        if nuc.shape!=prot.shape: return None,'Source channel shapes differ'
        labels=None
        if mask.is_file():
            with tifffile.TiffFile(mask) as tf:
                labels=tf.pages[z].asarray()
        if labels is not None and labels.shape!=nuc.shape: return None,'Source label shape does not match image'
        return (nuc,prot,labels,z),None
    except Exception as exc:
        return None,f'Source TIFF read failed: {type(exc).__name__}: {exc}'


def representative_image_lvilymph(root,z=7):
    path=root/'data/inspect/LVILymph/LVILymphBreastCancer_SIMILARITY10_In115.zip'
    mask=root/'data/inspect/LVILymph/full_mask_final_segmentation_hwatershed_bg500_90%.tif'
    if not path.is_file(): return None,'LVILymph aligned ZIP not found'
    try:
        import tifffile
        with ZipFile(path) as arc:
            def one(ch):
                names=[n for n in arc.namelist() if re.search(rf'/({re.escape(ch)})/0{z+1:03d}_',n) and n.lower().endswith(('.tif','.tiff'))]
                if not names:
                    names=[n for n in arc.namelist() if f'/{ch}/' in n and Path(n).name.startswith(f'{z+1:04d}_') and n.lower().endswith(('.tif','.tiff'))]
                if len(names)!=1: raise ValueError(f'Expected one section {z+1} for {ch}; found {len(names)}')
                return tifffile.imread(io.BytesIO(arc.read(names[0])))
            nuc=one('Ir193'); prot=one('Gd156')
        labels=None
        if mask.is_file():
            with tifffile.TiffFile(mask) as tf:labels=tf.pages[z].asarray()
        if nuc.shape!=prot.shape or (labels is not None and labels.shape!=nuc.shape):
            return None,'LVILymph channel/mask dimensions differ'
        return (nuc,prot,labels,z),None
    except Exception as exc:
        return None,f'LVILymph read failed: {type(exc).__name__}: {exc}'


def composite(nuc,prot,lab=None):
    def norm(a):
        a=np.log1p(np.maximum(a.astype(float),0))
        lo,hi=np.quantile(a,[.10,.997])
        return np.clip((a-lo)/max(hi-lo,1e-7),0,1)**.85
    n=norm(nuc); p=norm(prot)
    rgb=np.stack([np.clip(.17*n+.97*p,0,1), np.clip(.65*n+.06*p,0,1),
                 np.clip(.96*n+.52*p,0,1)],axis=-1)
    # Only use a sparse, 2D 4-neighbor region boundary for display.
    if lab is not None:
        edge=np.zeros(lab.shape,bool)
        edge[1:,:]|=(lab[1:,:]!=lab[:-1,:]) & (lab[1:,:]!=0)
        edge[:,1:]|=(lab[:,1:]!=lab[:,:-1]) & (lab[:,1:]!=0)
        rgb[edge]=.65*rgb[edge]+.35*np.array([.98,.93,.32])
    return rgb


def fig1(root,out):
    fig=plt.figure(figsize=(12.8,7.4))
    grid=fig.add_gridspec(2,2,height_ratios=[1.04,1.0],wspace=.15,hspace=.27)
    ax=fig.add_subplot(grid[0,:]);ax.set_xlim(0,12);ax.set_ylim(0,3.3);ax.axis('off');panel(ax,'A')
    nodes=[(0.4,2.00,2.2,'3D IMC sources','SecondHer2 / LVILymph'),
       (2.95,2.00,2.2,'Fixed 3D objects','Source pseudo-instances'),
       (5.50,2.00,2.2,'Hold out middle Z','Signal only; geometry known'),
       (8.05,2.00,2.2,'Sampling baselines','Nearest-1 / 2 / all'),
       (10.25,.55,1.45,'Risk transfer','Source-trained model')]
    for x,y,w,title,subtitle in nodes:
        ax.add_patch(patches.FancyBboxPatch((x,y),w,.89,boxstyle='round,pad=.10',
          edgecolor='#B8CBD3',facecolor=PALE,linewidth=1.1))
        ax.text(x+w/2,y+.58,title,ha='center',va='center',color=DARK,fontsize=10,weight='bold')
        ax.text(x+w/2,y+.23,subtitle,ha='center',va='center',color=GRAY,fontsize=7.3)
    for x in [2.65,5.2,7.75]:ax.annotate('',xy=(x+.27,2.44),xytext=(x-.07,2.44),arrowprops=dict(arrowstyle='->',color=BLUE,lw=1.4))
    ax.annotate('',xy=(10.99,1.54),xytext=(10.1,2.32),arrowprops=dict(arrowstyle='->',color=BLUE,lw=1.4))
    ax.text(.6,.8,'92 sections · 31 channels · n=2,914 eligible',color=BLUE,fontsize=10,weight='bold')
    ax.text(.6,.4,'16 registered sections · 29 channels · n=1,780 eligible',color=ORANGE,fontsize=10,weight='bold')
    info=[]
    for axis,kind,title in [(fig.add_subplot(grid[1,0]),'source','B  SecondHer2 / Gd156'),
                             (fig.add_subplot(grid[1,1]),'target','C  LVILymph / Gd156')]:
        if kind=='source': val,err=representative_image_secondher2(root)
        else: val,err=representative_image_lvilymph(root)
        if val is None:
            axis.set_facecolor('#F4F7F8');axis.set_xticks([]);axis.set_yticks([])
            axis.text(.5,.55,'IMAGE PANEL NOT READY',ha='center',va='center',transform=axis.transAxes,
                      color='#8E5B43',weight='bold',fontsize=13)
            axis.text(.5,.38,err[:100],ha='center',va='center',transform=axis.transAxes,color=GRAY,fontsize=8,wrap=True)
            info.append({'panel':kind,'status':'MISSING_REQUIRES_QC','reason':err})
        else:
            nuc,pr,lab,z=val
            axis.imshow(composite(nuc,pr,lab),interpolation='nearest')
            axis.set_xticks([]);axis.set_yticks([])
            axis.text(.01,.02,f'Plane {z+1}; magenta: Gd156; blue: Ir193; yellow: mask edges',
                transform=axis.transAxes,fontsize=7,color='white',ha='left',va='bottom',
                bbox=dict(facecolor='black',alpha=.60,edgecolor='none',pad=3))
            info.append({'panel':kind,'status':'IMAGE_RENDERED_NEEDS_MANUAL_REGISTRATION_QC',
                         'plane_number_one_based':z+1,'image_size_xy':[int(nuc.shape[1]),int(nuc.shape[0])]})
        axis.set_title(title,loc='left',color=DARK)
    fig.suptitle('Study design and source measurement data',fontsize=17,weight='bold',color=DARK,y=.99)
    save(fig,out,'Figure1_study_design_and_imaging')
    return info


def fig2(out,d):
    dd=d['lag']; p=dd[(dd.group=='primary_retained_clean') & (dd.z_lag<=4)].copy()
    fig,axs=plt.subplots(1,2,figsize=(12.8,4.65),gridspec_kw={'width_ratios':[1.04,.96]})
    if p.empty: raise ValueError('Empty primary Z lag data')
    for marker,color in [('HER2_3B5',BLUE),('HER2_29D8',ORANGE)]:
        g=p[p.marker==marker].sort_values('z_lag')
        axs[0].plot(g.physical_lag_um,g.median_single_plane_error,'o-',lw=2,ms=6,
                    label=M[marker],color=color)
        axs[1].plot(g.physical_lag_um,g.median_paired_extra_error_vs_lag1,'o-',lw=2,ms=6,
                    label=M[marker],color=color)
        for r in g.itertuples():
            axs[0].annotate(f'n={int(r.n_objects)}',(r.physical_lag_um,r.median_single_plane_error),
                      xytext=(0,8),textcoords='offset points',ha='center',color=color,fontsize=7)
    axs[0].set(xlabel='Z distance (µm; SecondHer2 nominal spacing)',
               ylabel='Median single-plane absolute log1p error',title='Error vs Z distance (changing population)')
    axs[1].axhline(0,color=GRAY,lw=.9)
    axs[1].set(xlabel='Z distance (µm)',ylabel='Median within-object extra error vs lag 1',
               title='Within-object matched comparison')
    for ax,name in zip(axs,'AB'):
        panel(ax,name);ax.set_xticks([2,4,6,8]);ax.grid(axis='y');ax.legend(loc='upper left',frameon=False)
    fig.text(.01,.01,'Lag 3+ includes progressively fewer long-lived pseudo-instances; do not infer a universal physical decay length.',
             fontsize=9,color='#8B4F29')
    fig.suptitle('Z-lag sensitivity of held-out protein intensity',fontsize=15,color=DARK,weight='bold')
    fig.tight_layout(rect=[0,.035,1,.93],w_pad=2.7)
    save(fig,out,'Figure2_z_distance_error')


def fig3(out,d):
    ss=d['source_sampling'];ts=d['target_sampling'];ci=d['target_sampling_ci']
    fig=plt.figure(figsize=(13,8.0));grid=fig.add_gridspec(2,2,hspace=.46,wspace=.29)
    ax=fig.add_subplot(grid[0,0]);panel(ax,'A')
    smethods=['nearest1','nearest2','all_others']
    xs=np.arange(2)
    for i,met in enumerate(smethods):
        vals=[]
        for marker in ['HER2_3B5','HER2_29D8']:
            q=ss[(ss.marker==marker)&(ss.method==met)]
            if len(q)!=1:raise ValueError(f'Source method missing: {marker}/{met}')
            vals.append(float(q.median_abs_log_error.iloc[0]))
        ax.bar(xs+(i-1)*.23, vals,width=.20,label=met.replace('_',' ').title(),color=C[met])
    ax.set(xticks=xs,xticklabels=['HER2 3B5','HER2 29D8'],ylabel='Median absolute log1p error',
            title='SecondHer2: held-out middle Z (n=2,914)');ax.legend(frameon=False,fontsize=8);ax.grid(axis='y')
    ax=fig.add_subplot(grid[0,1]);panel(ax,'B')
    tm=['Gd156','Cd112','Eu151','Yb174','Er168']
    rows=[]
    for t in tm:
        q=ts[ts.isotope==t]
        if len(q)!=1:raise ValueError(f'Target summary missing: {t}')
        rows.append(q.iloc[0])
    for i,met in enumerate(smethods):
        vals=[r[f'{met}_median_abs_log_error'] for r in rows]
        ax.bar(np.arange(len(tm))+(i-1)*.23,vals,width=.20,color=C[met],label=met)
    ax.set(xticks=np.arange(5),xticklabels=tm,ylabel='Median absolute log1p error',
           title='LVILymph: independent volume (n=1,780)');ax.grid(axis='y')
    ax=fig.add_subplot(grid[1,0]);panel(ax,'C')
    for i,marker in enumerate(['HER2_3B5','HER2_29D8']):
        q=ss[(ss.marker==marker)&(ss.method=='nearest1')]
        rr=q.iloc[0]
        # improvement(nearest1 vs nearest2) is NEGATIVE, so flip endpoints for nearest2 vs nearest1
        delta=-float(rr.mean_paired_improvement_vs_nearest2)
        low=-float(rr.bootstrap_upper_975);up=-float(rr.bootstrap_lower_025)
        ax.errorbar(delta,i,xerr=[[delta-low],[up-delta]],fmt='o',color=[BLUE,ORANGE][i],capsize=4)
    ax.axvline(0,color=GRAY,lw=.8)
    ax.set(yticks=[0,1],yticklabels=['HER2 3B5','HER2 29D8'],
           xlabel='Mean error reduction: Nearest-2 − Nearest-1 (positive = benefit)',
           title='SecondHer2: spatial-block intervals')
    ax.set_ylim(1.5,-.5);ax.grid(axis='x')
    ax=fig.add_subplot(grid[1,1]);panel(ax,'D')
    for i,t in enumerate(tm):
        q=ci[(ci.isotope==t)&(ci.method_a=='nearest2')&(ci.method_b=='nearest1')]
        if len(q)!=1:raise ValueError(f'Target paired comparison missing: {t}/nearest2/nearest1')
        row=q.iloc[0];delta=float(row.mean_error_improvement_a_vs_b)
        low=float(row.boot_p025);up=float(row.boot_p975)
        ax.errorbar(delta,i,xerr=[[delta-low],[up-delta]],fmt='o',color=TEAL,capsize=3)
    ax.axvline(0,color=GRAY,lw=.8)
    ax.set(yticks=np.arange(5),yticklabels=tm,
           xlabel='Mean error reduction: Nearest-2 vs Nearest-1',
           title='LVILymph: paired spatial-block intervals')
    ax.set_ylim(4.5,-.5);ax.grid(axis='x')
    fig.suptitle('Within-object local sampling replicates across two 3D volumes',
                 fontsize=15,weight='bold',color=DARK)
    fig.text(.02,.012,'Bar heights are medians. Forest-plot deltas and intervals use PAIRED MEAN errors. Spatial blocks are not biological replicates.',
             color='#86502E',fontsize=9)
    fig.subplots_adjust(top=.90,bottom=.115,left=.12,right=.97)
    save(fig,out,'Figure3_cross_volume_sampling')


def fig4(out,d):
    oo=d['source_risk'];ext=d['target_risk']; ci=d['target_risk_ci']
    fig=plt.figure(figsize=(12.8,7.5));grid=fig.add_gridspec(2,2,wspace=.31,hspace=.52)
    ax=fig.add_subplot(grid[0,0]);panel(ax,'A')
    methods=['variation','signal_only','full']
    for j,met in enumerate(methods):
        vals=[]
        for sm in ['HER2_3B5','HER2_29D8']:
            q=oo[(oo.source_model==sm)&(oo.method==met)]
            if len(q)!=1:raise ValueError(f'Source OOF row missing {sm}/{met}')
            vals.append(q.source_spatial_oof_auroc.iloc[0])
        ax.bar(np.arange(2)+(j-1)*.24,vals,width=.21,color=C[met],label=met)
    ax.set(xticks=[0,1],xticklabels=['HER2 3B5','HER2 29D8'],ylim=(.45,.7),
           ylabel='Spatial OOF AUROC',title='Source-only spatial validation')
    ax.legend(frameon=False);ax.grid(axis='y')
    ax=fig.add_subplot(grid[0,1]);panel(ax,'B')
    p=ext[(ext.source_model=='HER2_3B5')&(ext.target_marker=='Gd156')]
    for j,met in enumerate(methods):
        q=p[p.method==met]
        if len(q)!=1:raise ValueError(f'Primary target risk missing {met}')
        r=q.iloc[0];ax.bar(j,float(r.external_auroc_target_q75),color=C[met],width=.6)
        ax.text(j,float(r.external_auroc_target_q75)+.015,f'{r.external_auroc_target_q75:.3f}',ha='center',fontsize=9)
    ax.set(xlim=(-.65,2.65),ylim=(0,1.0),xticks=np.arange(3),xticklabels=['Variation','Signal-only','Full'],
           ylabel='Target high-error AUROC',title='Frozen 3B5 model → LVILymph Gd156')
    ax.grid(axis='y')
    ax=fig.add_subplot(grid[1,:]);panel(ax,'C')
    targets=['Gd156','Cd112','Eu151','Yb174','Er168']
    for j,met in enumerate(methods):
        vals=[]
        for target in targets:
            q=ext[(ext.source_model=='HER2_3B5')&(ext.target_marker==target)&(ext.method==met)]
            if len(q)!=1:raise ValueError(f'Target model missing: {target}/{met}')
            vals.append(float(q.external_auroc_target_q75.iloc[0]))
        ax.plot(targets,vals,marker=['s','o','^'][j],lw=2,markersize=7,color=C[met],label=met)
    ax.axhline(.5,color=GRAY,ls='--',lw=.8)
    ax.set(ylim=(.42,.87),ylabel='Within-target q75 high-error AUROC',
           title='Exploratory marker-specific transfer within one LVILymph volume')
    ax.grid(axis='y');ax.legend(frameon=False,ncol=3)
    fig.suptitle('Cross-volume protein measurement risk prediction',fontsize=15,weight='bold',color=DARK)
    fig.text(.015,.013,'The target errors are held-out Z-plane proxies; 5 markers belong to ONE target volume, not 5 biological samples.',
             color='#86502E',fontsize=9)
    fig.subplots_adjust(top=.9,bottom=.115,left=.11,right=.97)
    save(fig,out,'Figure4_reliability_transfer')


def fig5(out,d):
    ext=d['target_risk'];ci=d['target_risk_ci'];shift=d['domain_shift']
    fig=plt.figure(figsize=(13,7.8));grid=fig.add_gridspec(2,2,hspace=.50,wspace=.42)
    primary=ext[(ext.source_model=='HER2_3B5')&(ext.target_marker=='Gd156')]
    ax=fig.add_subplot(grid[0,0]);panel(ax,'A')
    methods=['variation','geometry','signal_only','full',
             'signal_no_absolute_intensity','full_no_absolute_intensity']
    labs=['Variation','Geometry','Signal-only','Full','Signal − intensity','Full − intensity']
    for i,met in enumerate(methods):
        q=primary[primary.method==met]
        if len(q)!=1:raise ValueError(f'Missing primary ablation: {met}')
        v=float(q.external_auroc_target_q75.iloc[0])
        ax.barh(i,v,color=C[met],height=.68)
        ax.text(v+.007,i,f'{v:.3f}',va='center',fontsize=8)
    ax.set(yticks=np.arange(len(methods)),yticklabels=labs,xlim=(0,1),
          xlabel='External AUROC (higher is better)',title='Primary target: feature ablation')
    ax.invert_yaxis();ax.grid(axis='x')
    ax=fig.add_subplot(grid[0,1]);panel(ax,'B')
    q=ci[(ci.source_model=='HER2_3B5')&(ci.target_marker=='Gd156')]
    contrasts=[('full-minus-signal_only','Full − Signal-only'),
               ('full-minus-full_no_absolute_intensity','Full − no intensity'),
               ('full_no_absolute_intensity-minus-signal_no_absolute_intensity','Geometry after intensity removal')]
    for i,(tag,title) in enumerate(contrasts):
        r=q[q.contrast==tag]
        if len(r)!=1:raise ValueError(f'Missing paired contrast {tag}')
        rr=r.iloc[0];delta=float(rr.delta_auroc_target_q75);lo=float(rr.lower_025);up=float(rr.upper_975)
        ax.errorbar(delta,i,xerr=[[delta-lo],[up-delta]],fmt='o',color=[BLUE,ORANGE,RED][i],capsize=4)
    ax.axvline(0,color=GRAY,lw=.8)
    ax.set(yticks=np.arange(3),yticklabels=[label for _,label in contrasts],
           xlabel='Paired difference in AUROC (95% block interval)',
           title='Evidence for geometry / intensity dependence')
    ax.set_ylim(2.6,-.6);ax.grid(axis='x')
    ax=fig.add_subplot(grid[1,0]);panel(ax,'C')
    for j,met in enumerate(['signal_only','full']):
        values=[]
        for t in ['Gd156','Cd112','Eu151','Yb174','Er168']:
            x=ext[(ext.source_model=='HER2_3B5')&(ext.target_marker==t)&(ext.method==met)]
            values.append(float(x.selective_risk_reduction_fraction.iloc[0]))
        ax.plot(['Gd156','Cd112','Eu151','Yb174','Er168'],np.array(values)*100,'o-',
                color=C[met],label=met,lw=1.9,markersize=6)
    ax.axhline(0,color=GRAY,lw=.8)
    ax.set(ylabel='Mean error reduction at 50% coverage (%)',
           title='Selective measurement error: marker dependence')
    ax.legend(frameon=False);ax.grid(axis='y')
    ax=fig.add_subplot(grid[1,1]);panel(ax,'D')
    ds=shift[(shift.source_model=='HER2_3B5')&(shift.target_marker=='Gd156')]
    if ds.empty: raise ValueError('Gd156 feature shift rows not found')
    ds=ds.sort_values('target_fraction_outside_source_95pct_range',ascending=True)
    label=ds.feature.str.replace('log_','log ',regex=False).str.replace('_',' ',regex=False)
    ax.barh(label,ds.target_fraction_outside_source_95pct_range*100,color=[RED if f=='log_nearest2_mean' else '#8CB1C5' for f in ds.feature])
    ax.set(xlim=(0,102),xlabel='Target outside source 95% feature range (%)',
           title='Input domain shift (source 3B5 → LVILymph Gd156)')
    ax.grid(axis='x')
    fig.suptitle('Ablations reveal limitations of portable risk models',fontsize=15,weight='bold',color=DARK)
    fig.text(.015,.007,'Domain shift and q75 labels limit generalization claims; geometry did not show a stable independent benefit.',
             color='#86502E',fontsize=9)
    fig.subplots_adjust(left=.15,right=.97,bottom=.115,top=.90)
    save(fig,out,'Figure5_ablation_and_domain_shift')


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def write_docs(out,image_status):
    legend='''# Manuscript main-figure draft legends\n\nThese captions are **drafts based on the frozen CSV results**. Numeric findings must be checked against the rendered panels. All confidence intervals are within-volume spatial-block resampling intervals and do not establish between-patient generalization.\n\n## Figure 1. Study design and 3D IMC source measurements\n(A) Study design for fixed three-dimensional pseudo-instances, leave-one-middle-Z signal prediction and source-only risk model transfer. (B) SecondHer2 composite: Ir193 nuclear channel in blue, Gd156 signal in magenta, source pseudoinstance boundaries in yellow, selected section. (C) LVILymph composite with corresponding isotope channels. Channels are dataset-specific; Gd156 is provisionally assigned c-erbB- in LVILymph; **antigen and sample provenance must be confirmed**. Panels (B-C) need visual registration QC before submission. Target protein signal is held out, but object geometry is derived from the existing full-volume segmentation.\n\n## Figure 2. Z-distance and held-out measurement error in SecondHer2\n(A) Median absolute log1p single-plane prediction error versus Z lag for HER2 3B5 and HER2 29D8, annotated with numbers of eligible pseudo-instances. (B) Median within-object difference relative to lag 1, restricting to matched objects at each lag. Numbers decrease at longer lags; the curves do not define a universal tissue correlation length.\n\n## Figure 3. Cross-volume evaluation of Z sampling policies\n(A-B) Held-out middle-Z median absolute log1p error for Nearest-1, Nearest-2 and all remaining sections in SecondHer2 and LVILymph respectively. (C-D) Paired **mean** error reduction for Nearest-2 relative to Nearest-1, with spatial-block bootstrap 95% quantile intervals. Median bars and mean comparisons are different estimands. Independent volumes are not proven independent patients.\n\n## Figure 4. Portable measurement risk model transfer\n(A) Three-fold spatial out-of-fold AUROC in SecondHer2 source pseudo-instances. (B) Frozen source HER2 3B5 model evaluated on LVILymph Gd156 for predicting upper-quartile held-out Z-plane measurement error. (C) Exploratory cross-marker transfer of the source-trained model to five protein channels in one LVILymph volume. Evaluation labels use each target marker’s within-volume q75; no LVILymph labels are used for model fitting.\n\n## Figure 5. Feature ablation and cross-volume limitations\n(A) Comparison of variation baseline, geometry-only, signal-only and combined models with/without explicit log absolute intensity. (B) Paired block-bootstrap AUROC contrasts in the primary cross-volume mapping. (C) Conditional selective error reduction when retaining the lowest-risk half of pseudo-instances by marker. (D) Feature shift between source and target distributions; a major shift in absolute intensity limits unqualified transfer claims.\n\n## Submission gate (must review before journal submission)\n- Confirm LVILymph Gd156 antigen identity against full antibody panel, not the truncated `c-erbB-` label.\n- Confirm physical Z spacing, registration, 16-of-17 section selection and biological sample independence.\n- Inspect both Figure 1 images at full resolution; matched image-mask dimensions alone are insufficient.\n- Cite original 3D IMC acquisition and segmentation methods; source IDs are *pseudo-instances*, not independent manual cell ground truth.\n- Distinguish within-target q75 proxy risk AUROC from fixed-source absolute error thresholds.\n- Treat marker-level outcomes from one volume as correlated; do not imply five patient-level replications.\n- Report unsuccessful hotspot filtering and geometric ablations transparently in supplementary results.\n'''
    (out/'MAIN_FIGURE_LEGENDS_DRAFT.md').write_text(legend,encoding='utf8')
    index='''# Main-figure assembly / evidence map\n\n| Figure | Claim it can support | CSV evidence | Gate |\n|---|---|---|---|\n| Figure 1 | Study design and actual isotope images | Original OME and aligned TIFF, original masks | Check antigen identity and overlay registration manually |\n| Figure 2 | Within-SecondHer2 Z-lag sensitivity | `z_lag_error_summary.csv` | Declining cohort size with lag |\n| Figure 3 | Nearest-2 vs Nearest-1 in both volumes; all-others variable | `core_comparison_summary.csv`, `external_sampling_summary.csv`, `paired_spatial_block_intervals.csv` | Do not equate medians with paired mean differences |\n| Figure 4 | Cross-volume heldout measurement risk ranking | `source_ablation_spatial_oof.csv`, `external_ablation_summary.csv` | Only one target volume; conditional proxy labels |\n| Figure 5 | Geometry non-benefit, intensity dependence, marker heterogeneity | `external_paired_spatial_block_intervals.csv`, `feature_domain_shift.csv` | Strong OOD covariate shift |\n\n**Supplementary candidates:** 3D pseudo-instance HER2 rank-discordance map and Moran’s I sensitivity; fixed-boundary method comparison; failed hotspot-risk filter; negative kernel experiments; source-mask traceability; method hyperparameters and cohort audit. Do not claim 3D universally beats genuine 2D tissue acquisition: projection-reference experiment reverses the method ranking.\n\nAll figures are provisional, generated from existing files without model retraining. No invented measurements were inserted.\n'''
    (out/'MAIN_FIGURE_INDEX.md').write_text(index,encoding='utf8')
    (out/'FIGURE1_IMAGE_STATUS.json').write_text(json.dumps(image_status,indent=2,ensure_ascii=False),encoding='utf8')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root',type=Path,default=Path('.'))
    ap.add_argument('--out',type=Path,default=Path('releases/3d_manuscript_evidence_v0.1/main_figures_v0.1'))
    args=ap.parse_args()
    root=args.project_root.resolve();out=(root/args.out).resolve()
    if root not in out.parents: raise ValueError('Output must stay under project root')
    if out.exists(): raise FileExistsError(f'Refusing to overwrite existing figure edition: {out}')
    print('[1/6] Read and validate eight frozen result tables',flush=True)
    data=load_checked(root)
    setup_style()
    out.mkdir(parents=True)
    print('[2/6] Figure 1: study design and optional actual tissue images',flush=True)
    status=fig1(root,out)
    print('[3/6] Figure 2–3: Z lag and paired cross-volume sampling',flush=True)
    fig2(out,data);fig3(out,data)
    print('[4/6] Figure 4–5: source-to-target risk transfer and ablations',flush=True)
    fig4(out,data);fig5(out,data)
    print('[5/6] Write captions, evidence map and hashes',flush=True)
    write_docs(out,status)
    audit=[]
    for name,rel in INPUTS.items():
        path=root/rel
        audit.append({'key':name,'relative_path':rel,'size_bytes':path.stat().st_size,'sha256':sha256(path),'n_rows':len(data[name])})
    pd.DataFrame(audit).to_csv(out/'FIGURE_INPUT_SHA256_MANIFEST.csv',index=False)
    (out/'RENDER_PROTOCOL.json').write_text(json.dumps({
       'version':'main_figures_v0.1', 'source':'frozen local experimental CSVs',
       'figures':[f'Figure{i}' for i in range(1,6)],
       'images':status,'n_input_csv':len(audit),'rendered_pdf_png':True,
       'requires_manual_qc':True,'not_independent_biological_validation':True,
       'not_new_model_training':True,
       'figure2_lag_not_directly_comparable_to_nearest1_policy':True,
       'source_target_volume_independence_not_patient_independence':True},indent=2),encoding='utf8')
    print('[6/6] Final output audit',flush=True)
    names=sorted(p.name for p in out.glob('Figure*.pdf'))
    if len(names)!=5:raise RuntimeError(f'Expected 5 figure PDFs, got {names}')
    print('SUCCESS:',out)
    for p in names:print('  ',p)
    print('Image panel status:',', '.join(x['panel']+':'+x['status'] for x in status))
    print('NOTE: All figures require author QC; no biological truth is implied.',flush=True)

if __name__=='__main__':main()
