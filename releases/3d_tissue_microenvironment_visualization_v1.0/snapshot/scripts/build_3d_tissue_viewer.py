#!/usr/bin/env python3
"""Workflow 10 pilot: local offline 3D pseudo-instance centroid and voxel visualization.

XY is in pixel indices, Z in slice indices. No physical 3D tissue geometry claim.
"""
from __future__ import annotations
import argparse, hashlib, json, shutil, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile

MARKERS=['Ir193','panCK','HER2','CD31','LYVE1']

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for ch in iter(lambda:f.read(4*1024*1024),b''):h.update(ch)
    return h.hexdigest()

def validate(data,summary):
    expected={'instance_id','centroid_y_px','centroid_x_px','z_span_planes','qc_flag_review',
              'voxel_count','n_xy_radius_neighbors'}|{'mean_raw_'+m for m in MARKERS}
    missing=expected-set(data.columns)
    if missing:raise ValueError('Missing columns: '+str(sorted(missing)))
    if data.instance_id.duplicated().any() or data.instance_id.isna().any():raise ValueError('Nonunique instance ID')
    if len(data)!=summary['n_instances']:raise ValueError('Instance count differs from verified Workflow09')
    for field in expected - {'qc_flag_review'}:
        if not np.isfinite(pd.to_numeric(data[field],errors='coerce').to_numpy(dtype=float)).all():
            raise ValueError('Missing or non-finite data: '+field)
    if set(summary.get('marker_names',[]))!=set(MARKERS):raise ValueError('Marker panel changed')

def scene(fig,focus=False):
    fig.update_layout(template='plotly_white',height=800,
                      margin=dict(l=0,r=0,t=95,b=0),
                      scene=dict(xaxis=dict(title='X pixel'),yaxis=dict(title='Y pixel'),
                                 zaxis=dict(title='Z section index — NOT µm'),
                                 aspectmode='manual',
                                 aspectratio=dict(x=1.3,y=1.3,z=.5 if focus else .38)),
                      annotations=[dict(text='PSEUDO-INSTANCES · Z=index only · visual alignment plausible, not independently registered',
                                        x=.5,y=1.02,xref='paper',yref='paper',showarrow=False,
                                        font=dict(size=12,color='#8b3a27'))])

def make_overview(df):
    import plotly.graph_objects as go
    ids=df.instance_id.to_numpy(dtype=int)
    flagged=df.qc_flag_review.astype(str).str.lower().isin(['true','1']).to_numpy()
    z=(df['z_span_planes'].astype(float).to_numpy()*0) # overwritten below via provided centroid_z
    # centroid_z_index from spatial integration was not carried forward; infer midpoint from morphometry separately.
    z=df['centroid_z_index'].to_numpy(float)
    x=df.centroid_x_px.to_numpy(float);y=df.centroid_y_px.to_numpy(float)
    hover=np.stack([ids,df.voxel_count.to_numpy(int),df.z_span_planes.to_numpy(int),
                    df.n_xy_radius_neighbors.to_numpy(int),flagged.astype(int)],axis=1)
    fig=go.Figure()
    for index,m in enumerate(MARKERS):
        raw=df['mean_raw_'+m].to_numpy(float)
        clip=float(np.quantile(raw,.99))
        upper=max(clip,1e-6)
        # Keep raw values in tooltip and log-map to suppress a few extreme points.
        intensity=np.log1p(np.clip(raw,0,upper))
        custom=np.column_stack([hover,raw])
        fig.add_trace(go.Scatter3d(name=m,x=x,y=y,z=z,mode='markers',visible=(index==0),
                                   marker=dict(size=2.4,opacity=.78,color=intensity,
                                               colorscale='Viridis',cmin=0,cmax=np.log1p(upper),
                                               colorbar=dict(title='log1p(clip mean '+m+')')),
                                   customdata=custom,
                                   hovertemplate='ID=%{customdata[0]}<br>X=%{x:.1f}, Y=%{y:.1f}, Z=%{z:.1f}<br>'
                                                 'voxels=%{customdata[1]}<br>Zspan=%{customdata[2]}<br>'
                                                 'XY neighbors=%{customdata[3]}<br>QC flag=%{customdata[4]}<br>'
                                                 +m+' raw mean=%{customdata[5]:.4f}<extra></extra>'))
    fig.add_trace(go.Scatter3d(name='QC boundary/review',x=x,y=y,z=z,mode='markers',visible=False,
                               marker=dict(size=2.0,opacity=.65,color=flagged.astype(int),
                                           colorscale=[[0,'#2d7db0'],[.5,'#2d7db0'],[.5+1e-6,'#e06a3a'],[1,'#e06a3a']],
                                           cmin=0,cmax=1,showscale=False),
                               customdata=hover,
                               hovertemplate='ID=%{customdata[0]}<br>QC flag=%{customdata[4]}<extra></extra>'))
    opts=[]
    for i,m in enumerate(MARKERS+['QC flag']):
        opts.append(dict(label=m,method='update',args=[{'visible':[j==i for j in range(len(MARKERS)+1)]},
                                                       {'title':f'Workflow 10 | 3D pseudo-instance centroid map — {m}'}]))
    fig.update_layout(title='Workflow 10 | 3D pseudo-instance centroid map — Ir193',
                      updatemenus=[dict(buttons=opts,direction='down',x=0,xanchor='left',y=1.12,yanchor='top')])
    scene(fig)
    return fig

def densest_roi(mask,side):
    """Find dense XY window via integral image, with no padding or arbitrary relocation."""
    proj=np.count_nonzero(mask,axis=0).astype(np.int64)
    height,width=proj.shape
    side=min(int(side),height,width)
    if side<8:raise ValueError('ROI side must be >=8')
    ii=np.pad(proj,((1,0),(1,0))).cumsum(0).cumsum(1)
    windows=ii[side:,side:]-ii[:-side,side:]-ii[side:,:-side]+ii[:-side,:-side]
    y,x=np.unravel_index(int(np.argmax(windows)),windows.shape)
    return int(y),int(x),int(side),int(windows[y,x])

def make_roi(mask,df,roi_side=192,max_points=35000,marker='HER2',seed=42):
    import plotly.graph_objects as go
    y,x,side,nfg=densest_roi(mask,roi_side)
    cropped=mask[:,y:y+side,x:x+side]
    zs,ys,xs=np.nonzero(cropped)
    if not len(zs):raise ValueError('No instances in selected ROI')
    rng=np.random.default_rng(seed)
    selection=np.sort(rng.choice(len(zs),size=min(len(zs),max_points),replace=False))
    zs=zs[selection];ys=ys[selection]+y;xs=xs[selection]+x
    ident=cropped[zs,ys-y,xs-x].astype(np.int64)
    frame=df.set_index('instance_id')
    missing=set(np.unique(ident))-set(frame.index)
    if missing:raise ValueError(f'ROI contains unknown mask instance IDs: {list(missing)[:8]}')
    raw=frame['mean_raw_'+marker].reindex(ident).to_numpy(float)
    upper=float(np.quantile(df['mean_raw_'+marker].to_numpy(float),.99))
    value=np.log1p(np.clip(raw,0,max(upper,1e-6)))
    fig=go.Figure(go.Scatter3d(x=xs,y=ys,z=zs,mode='markers',
                   marker=dict(size=2.2,opacity=.48,color=value,colorscale='Viridis',
                               colorbar=dict(title='log1p('+marker+' mean)'),
                               cmin=0,cmax=np.log1p(max(upper,1e-6))),
                   customdata=np.column_stack([ident,raw]),
                   hovertemplate='Pseudo-instance ID=%{customdata[0]}<br>'
                                 'X=%{x},Y=%{y},Z=%{z}<br>'+marker+
                                 ' mask-restricted mean=%{customdata[1]:.4f}<extra></extra>'))
    fig.update_layout(title=f'Workflow 10 | 3D voxel sample in densest {side}×{side} XY ROI — {marker}')
    scene(fig,focus=True)
    return fig,{'roi_y0':y,'roi_x0':x,'roi_side_px':side,'foreground_voxels_in_roi':nfg,
                'visualized_voxels':len(zs),'color_marker':marker,'unique_visible_instance_ids':len(set(ident.tolist()))}

def make(mask_path,source,summary_path,out,roi_side=192,max_points=35000):
    if out.exists():raise FileExistsError('Viewer output already exists: '+str(out))
    summary=json.loads(summary_path.read_text())
    data=pd.read_csv(source)
    maskinfo=summary['source_mask_sha256']
    if sha(mask_path)!=maskinfo:raise RuntimeError('Source mask was modified after Workflow09 extraction')
    msummary=summary_path.parent.parent/'3d_cell_morphometry_v1'/'cell_morphometry.csv'
    if not msummary.exists():raise FileNotFoundError('Workflow07 centroid_z_index table missing: '+str(msummary))
    morph=pd.read_csv(msummary,usecols=['instance_id','centroid_z_index'])
    if morph.instance_id.duplicated().any():raise ValueError('Duplicate ID in Workflow07')
    if sha(source)!=json.loads((summary_path.parent/'output_file_sha256.json').read_text())['instance_marker_spatial_table.csv']:
        raise RuntimeError('Workflow09 spatial table modified')
    data=data.merge(morph,on='instance_id',how='left',validate='one_to_one')
    validate(data,summary)
    if data.centroid_z_index.isna().any():raise ValueError('Missing 3D centroid Z')
    mask=tifffile.imread(mask_path)
    if mask.ndim!=3 or list(mask.shape)!=summary['mask_shape_assumed_zyx']:
        raise ValueError('Mask shape does not match Workflow09')
    if set(np.unique(mask))- {0} != set(data.instance_id.to_numpy(int)):
        raise ValueError('Mask IDs differ from Workflow09 table')
    out.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.viewer10_tmp_',dir=out.parent))
    try:
        fig=make_overview(data)
        fig.write_html(stage/'centroid_marker_viewer.html',include_plotlyjs=True,full_html=True,
                       config={'responsive':True,'displaylogo':False})
        fig2,detail=make_roi(mask,data,roi_side,max_points)
        fig2.write_html(stage/'roi_voxel_viewer.html',include_plotlyjs=True,full_html=True,
                        config={'responsive':True,'displaylogo':False})
        provenance={'version':'0.1.0-exploratory-viewer','n_instances':len(data),
                    'mask_sha256':sha(mask_path),'spatial_table_sha256':sha(source),
                    'workflow09_summary_sha256':sha(summary_path),
                    'z_units':'integer slice index ONLY; physical spacing unverified',
                    'marker_units':'processed TIFF mean inside watershed pseudo-ID',
                    'real_data_has_one_specimen':True,'roi':detail,
                    'warning':'No biological cell-identity or direct molecular interaction conclusion.'}
        (stage/'viewer_summary.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
        digests={p.name:sha(p) for p in stage.iterdir() if p.is_file()}
        (stage/'viewer_sha256.json').write_text(json.dumps(digests,indent=2)+'\n')
        stage.rename(out)
    finally:
        if stage.exists():shutil.rmtree(stage)
    return provenance

def verify(out,source,mask_path,summary_path):
    declared=json.loads((out/'viewer_sha256.json').read_text())
    actual={p.name for p in out.iterdir() if p.is_file()}
    if actual!=set(declared)|{'viewer_sha256.json'}:raise RuntimeError('Missing/extra viewer files')
    for name,h in declared.items():
        if sha(out/name)!=h:raise RuntimeError('Viewer output changed: '+name)
    summary=json.loads((out/'viewer_summary.json').read_text())
    if sha(source)!=summary['spatial_table_sha256'] or sha(mask_path)!=summary['mask_sha256'] or sha(summary_path)!=summary['workflow09_summary_sha256']:
        raise RuntimeError('Upstream viewer input changed')
    return {'verified_viewer_files':len(declared),'n_instances':summary['n_instances']}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mask',type=Path,default=Path('data/inspect/LVILymph/full_mask_final_segmentation_hwatershed_bg500_90%.tif'))
    ap.add_argument('--table',type=Path,default=Path('runs/3d_protein_marker_integration_v1/instance_marker_spatial_table.csv'))
    ap.add_argument('--summary',type=Path,default=Path('runs/3d_protein_marker_integration_v1/marker_integration_summary.json'))
    ap.add_argument('--out',type=Path,default=Path('runs/3d_tissue_microenvironment_viewer_v1'))
    ap.add_argument('--roi-side',type=int,default=192)
    ap.add_argument('--max-points',type=int,default=35000)
    ap.add_argument('--verify',action='store_true')
    a=ap.parse_args()
    output=verify(a.out,a.table,a.mask,a.summary) if a.verify else make(a.mask,a.table,a.summary,a.out,a.roi_side,a.max_points)
    print(json.dumps(output,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
