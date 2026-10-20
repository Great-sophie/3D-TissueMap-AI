#!/usr/bin/env python3
"""Workflow 11I: blind, orthogonal-channel boundary review (exploratory, single specimen).

Prepare PNG tiles and a self-contained local annotation HTML, then evaluate visible
manual polylines against frozen Workflow 11H label volumes without touching inputs.

Reference channels: Ir193 nuclear, Cd112 panCK, Gd160 CD44 (CD44 is NOT universal
membrane ground truth). Both model outputs, Yb174 and HER2 are hidden from readers.

Only sparse-line coverage/distance are computed, not boundary F1 or Dice.
"""
import argparse
import hashlib
import io
import json
import re
import sys
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from scipy.ndimage import distance_transform_edt

CHANNELS = {"Ir193": "nuclear", "Cd112": "panCK", "Gd160": "CD44"}
SOURCE_DISPLAY = "Ir193 + panCK (Cd112) + CD44 (Gd160); NO Cadherin/HER2/model predictions"
DEFAULT_Z = [0, 5, 10, 15]
VERSION = "11I.0.1"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def members_for(zf, metal):
    pat = re.compile(rf"__(?:{re.escape(metal)})\.tiff?$", re.IGNORECASE)
    paths = [s for s in zf.namelist() if pat.search(Path(s).name)]
    def idx(path):
        match = re.search(r"^(\d+)_", Path(path).name)
        if match is None:
            raise RuntimeError(f"No numbered slice basename: {path}")
        return int(match.group(1))
    paths.sort(key=idx)
    numbers = [idx(p) for p in paths]
    if numbers != list(range(1, 17)):
        raise RuntimeError(f"{metal}: expected section numbers 0001..0016, got {numbers}")
    return paths


def to_gray(arr, scale):
    a = np.log1p(np.maximum(arr.astype(np.float64), 0))
    return (255 * np.clip(a / max(scale, 1e-6), 0, 1)).astype(np.uint8)


def _annotation_html(patches):
    patch_json = json.dumps(patches, ensure_ascii=False)
    html = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"/>
<title>Workflow 11I · Blinded Reference Annotation</title><style>
body{margin:0;font:15px system-ui,Arial;background:#121923;color:#eaf0fa}
header{padding:14px 22px;background:#253248;border-bottom:1px solid #556}
main{display:flex;gap:24px;margin:22px;flex-wrap:wrap}
section{max-width:560px}.panel{background:#1c2939;padding:16px;border-radius:9px}
canvas{border:2px solid #a3b4c7;cursor:crosshair;image-rendering:pixelated;width:480px;height:480px;max-width:90vw}
button,input{font:inherit;padding:8px;margin:4px;border-radius:6px}button{background:#296d9f;color:white;border:0;cursor:pointer}
button:disabled{opacity:.4}button.secondary{background:#46576b}button.warning{background:#9a5035}
small{color:#b8c9dd}.thumbs{display:flex;gap:8px}.thumbs img{width:112px;image-rendering:pixelated;border:1px solid #64748b}
.stat{margin:10px 0;color:#aee0bc}ul{line-height:1.65}li{margin:5px 0}
</style></head><body><header><b>Workflow 11I · Blinded orthogonal-channel boundary annotation</b><br>
<small>Ir193 + panCK + CD44 only. Neither Cadherin, HER2, nor predicted labels are visible.</small></header>
<main><section class="panel"><div id="patchTitle"></div>
<canvas id="canvas" width="80" height="80"></canvas>
<div class="thumbs"><span><img id="nuclear"><br><small>Ir193</small></span>
<span><img id="panCK"><br><small>panCK</small></span><span><img id="CD44"><br><small>CD44</small></span>
<span><img id="composite"><br><small>Composite</small></span></div>
<p><button onclick="finish('visible')">Save visible line</button>
<button class="warning" onclick="finish('uncertain')">Save uncertain line</button>
<button class="secondary" onclick="undo()">Undo vertex</button>
<button class="secondary" onclick="removeLast()">Delete last line</button></p>
<p><button onclick="previous()">◀ Previous</button><button onclick="next()">Next ▶</button></p>
<p><button onclick="exportJson()">Export annotation JSON</button>
<label style="display:block">Import saved annotation JSON: <input type="file" id="import" accept=".json"/></label></p>
<div class="stat" id="stats"></div></section>
<section class="panel"><h3>Annotation protocol</h3><p><label>Annotator ID (required):
<input type="text" id="annotator" value="reader01"/></label></p>
<ul><li>Zoomed composite at left; thumbnails show each reference channel.</li>
<li>Click sequential points <b>along a genuinely visible cell–cell interface</b>.</li>
<li>Click <b>Save visible line</b> to save a polyline (minimum 2 points).</li>
<li>If the boundary is unclear, save it as <b>uncertain</b>, or skip the patch.</li>
<li>Do not draw arbitrary nearest-neighbor lines between nuclei.</li>
<li>Keep annotations independent of both models; avoid looking at HER2 or Yb174 while annotating.</li>
<li>Every patch, including blank patches, stays in the review set; no cherry-picking successful patches.</li>
<li>Export JSON often. Import resumes from an earlier file. Your browser may not autosave.</li>
<li>Each point is stored in local pixel coordinates; the evaluator resolves Z and original image coordinates.</li>
</ul><p><small>These channels do not necessarily make every true membrane visible. A partially
annotated patch can support only boundary-line coverage/distance, not global precision, F1 or Dice.
One tissue specimen cannot establish independent biological generalization.</small></p>
<h3>Lines already recorded</h3><div id="lines"></div></section></main>
<script>const patches=__PATCH_JSON__; const PATCH_SET_TOKEN='__PATCH_TOKEN__'; let ix=0, lines=[], drawing=[], composite=new Image(), visited=new Set();
const canvas=document.getElementById('canvas'), ctx=canvas.getContext('2d');
function p(){return patches[ix]};
function render(){let patch=p(); visited.add(patch.id); composite.src=patch.images.composite;
for(let channel of ['nuclear','panCK','CD44','composite']) document.getElementById(channel).src=patch.images[channel];
canvas.width=patch.width; canvas.height=patch.height;
composite.onload=draw;
document.getElementById('patchTitle').textContent=`${ix+1}/${patches.length} · ${patch.id} · z=${patch.z} · x=${patch.x0}:${patch.x1} y=${patch.y0}:${patch.y1}`;
drawing=[]; draw(); }
function draw(){ctx.clearRect(0,0,canvas.width,canvas.height);
if(composite.complete&&composite.naturalWidth) ctx.drawImage(composite,0,0,canvas.width,canvas.height);
const selected=lines.filter(x=>x.patch_id===p().id);
function stroke(pts,color){if(!pts.length) return;ctx.beginPath();ctx.strokeStyle=color;ctx.lineWidth=.65;
ctx.moveTo(pts[0][0],pts[0][1]);for(let q of pts.slice(1))ctx.lineTo(q[0],q[1]);ctx.stroke();
for(let q of pts){ctx.beginPath();ctx.fillStyle=color;ctx.arc(q[0],q[1],.7,0,Math.PI*2);ctx.fill();}}
selected.forEach(x=>stroke(x.points_xy_local,x.category==='visible'?'#4dff89':'#ffc247'));
stroke(drawing,'#ff487b');
document.getElementById('stats').textContent=`Recorded: ${lines.length} lines (${selected.length} on this patch). Unfinished: ${drawing.length} points.`;
document.getElementById('lines').textContent=selected.map(x=>`${x.category}: ${x.points_xy_local.length} vertices`).join(' | ')||'No lines recorded on this patch';}
canvas.addEventListener('click',e=>{let r=canvas.getBoundingClientRect();
let x=(e.clientX-r.left)*canvas.width/r.width, y=(e.clientY-r.top)*canvas.height/r.height;
drawing.push([Math.min(canvas.width-1,Math.max(0,Number(x.toFixed(3)))),Math.min(canvas.height-1,Math.max(0,Number(y.toFixed(3))))]); draw();});
function finish(category){if(drawing.length<2){alert('Click at least 2 vertices first.');return;}
lines.push({patch_id:p().id,category,points_xy_local:drawing});drawing=[];draw();}
function undo(){drawing.pop();draw()};function removeLast(){for(let k=lines.length-1;k>=0;k--){if(lines[k].patch_id===p().id){lines.splice(k,1);break}}draw()};
function next(){if(ix<patches.length-1){ix++;render()}};function previous(){if(ix>0){ix--;render()}};
function exportJson(){let who=document.getElementById('annotator').value.trim();if(!who){alert('Enter annotator ID.');return;}
let data={schema_version:'11I.0.1',annotator_id:who,reference_channels:['Ir193','Cd112','Gd160'],blinding:'No Yb174/HER2/method masks',
patch_count:patches.length,patch_set_token:PATCH_SET_TOKEN,reviewed_patch_ids:Array.from(visited),lines};
let blob=new Blob([JSON.stringify(data,null,2)],{type:'application/json'});let a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=`workflow11i_${who}.json`;a.click();URL.revokeObjectURL(a.href)}
document.getElementById('import').addEventListener('change',async e=>{let f=e.target.files[0];if(!f)return;
try{let x=JSON.parse(await f.text());if(x.schema_version!=='11I.0.1')throw Error('Wrong schema');
if(!Array.isArray(x.lines))throw Error('Missing lines');if(x.patch_set_token!==PATCH_SET_TOKEN)throw Error('Patch set mismatch');lines=x.lines;visited=new Set(x.reviewed_patch_ids||[]);document.getElementById('annotator').value=x.annotator_id||'reader01';draw()}
catch(err){alert('Import failed: '+err.message)}});
render();</script></body></html>'''
    return html.replace('__PATCH_JSON__', patch_json).replace('__PATCH_TOKEN__', hashlib.sha256(patch_json.encode('utf-8')).hexdigest())


def prepare(args):
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(f"Output already exists: {out}; no overwrite allowed")
    if len(args.roi) != 4:
        raise ValueError("--roi expects x0 x1 y0 y1")
    x0, x1, y0, y1 = args.roi
    if x0 >= x1 or y0 >= y1 or args.tile <= 8:
        raise ValueError("Invalid ROI or tile size")
    if (x1 - x0) % args.tile or (y1 - y0) % args.tile:
        raise ValueError("ROI dimensions must be exact multiples of tile size")
    for z in args.z:
        if z < 0 or z >= 16:
            raise ValueError(f"Z out of range: {z}")
    selected = sorted(set(args.z))
    imgs = {}
    with zipfile.ZipFile(args.source_zip) as zf:
        paths = {metal: members_for(zf, metal) for metal in CHANNELS}
        for metal in CHANNELS:
            for z in selected:
                raw = tifffile.imread(io.BytesIO(zf.read(paths[metal][z])))
                if raw.ndim != 2 or raw.shape[0] < y1 or raw.shape[1] < x1:
                    raise ValueError(f"Shape/ROI mismatch: {paths[metal][z]}: {raw.shape}")
                imgs[(z, metal)] = raw[y0:y1, x0:x1].copy()
    scales = {metal: float(np.percentile(
        np.concatenate([np.log1p(imgs[(z,metal)].astype(np.float64)).ravel() for z in selected]),
        args.percentile)) for metal in CHANNELS}
    # write into staging directory, and rename only when complete
    stage = out.parent / ("." + out.name + "_staging")
    if stage.exists():
        raise FileExistsError(f"Old staging dir: {stage}")
    stage.mkdir(parents=True)
    try:
        (stage / 'patches').mkdir()
        patches = []
        for z in selected:
            allg = {metal: to_gray(imgs[(z,metal)], scales[metal]) for metal in CHANNELS}
            for yy in range(y0, y1, args.tile):
                for xx in range(x0, x1, args.tile):
                    sl = (slice(yy-y0, yy-y0+args.tile), slice(xx-x0,xx-x0+args.tile))
                    pid = f"z{z:02d}_y{yy:04d}_x{xx:04d}"
                    regions = {CHANNELS[m]: allg[m][sl] for m in CHANNELS}
                    comp = np.stack([regions['CD44'],regions['panCK'],regions['nuclear']], axis=2)
                    regions['composite'] = comp
                    rel = {}
                    for name, img in regions.items():
                        path = stage / 'patches' / f'{pid}_{name}.png'
                        Image.fromarray(img).save(path)
                        rel[name] = f'patches/{path.name}'
                    patches.append(dict(id=pid,z=int(z),x0=int(xx),y0=int(yy),x1=int(xx+args.tile),y1=int(yy+args.tile),
                                        width=int(args.tile),height=int(args.tile),images=rel))
        meta=dict(workflow='11I blinded sparse-line boundary review',version=VERSION,
            source_zip=str(args.source_zip),source_zip_sha256=sha256(args.source_zip),
            reference_channels=['Ir193','Cd112','Gd160'], display_notes=SOURCE_DISPLAY,
            no_prediction_labels_used_in_sampling=True,
            roi_xy=[x0,x1,y0,y1], z_indices=selected,tile=int(args.tile),percentile=float(args.percentile),
            patches=patches,patch_set_token=hashlib.sha256(json.dumps(patches,ensure_ascii=False).encode('utf-8')).hexdigest(),
            limitations=['CD44 is not pan-cell membrane truth','Reviewers must mark uncertain interfaces','Sparse polylines cannot support precision/F1/Dice','Only one specimen; Z planes are correlated'])
        (stage/'annotation_manifest.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')
        (stage/'ANNOTATION_GUIDE.txt').write_text(
            '1. Open annotate_blind.html in a normal web browser.\n'
            '2. View ONLY nuclear (Ir193), panCK (Cd112) and CD44 (Gd160) reference channels.\n'
            '3. Click at least two vertices along independently visible cell-cell interfaces.\n'
            '4. Save visible or uncertain. Do not guess where boundaries cannot be seen.\n'
            '5. Save annotation JSON and keep a copy. Prefer two independent annotators.\n'
            '6. Never reveal Cadherin / HER2 / predictions before independent annotation is complete.\n'
            '7. Evaluate only visible lines. Sparse-line matching is NOT full segmentation accuracy.\n',encoding='utf-8')
        (stage/'annotate_blind.html').write_text(_annotation_html(patches),encoding='utf-8')
        checks={str(p.relative_to(stage)):sha256(p) for p in stage.rglob('*') if p.is_file()}
        (stage/'SHA256SUMS.json').write_text(json.dumps(checks, indent=2),encoding='utf-8')
        stage.rename(out)
        print(json.dumps(dict(status='PASS',mode='prepare',patches=len(patches),z=selected,
            displayed_channels=list(CHANNELS),out=str(out)),indent=2))
    except Exception:
        import shutil
        shutil.rmtree(stage)
        raise


def verify(args):
    p=Path(args.out)
    meta=json.loads((p/'annotation_manifest.json').read_text(encoding='utf-8'))
    checks=json.loads((p/'SHA256SUMS.json').read_text(encoding='utf-8'))
    for name, digest in checks.items():
        path=p/name
        if not path.is_file() or sha256(path)!=digest:
            raise RuntimeError(f'Integrity failure: {path}')
    # File inventory strict
    actual={str(q.relative_to(p)) for q in p.rglob('*') if q.is_file()}
    if actual != set(checks)|{'SHA256SUMS.json'}:
        raise RuntimeError('Unexpected or missing files inside locked annotation package')
    print(json.dumps(dict(status='PASS',mode='verify',verified_files=len(checks),patches=len(meta['patches'])),indent=2))


def shared_interface(labels):
    """Only interfaces between two positive, unequal instance labels. No tissue rim."""
    a=np.asarray(labels)
    if a.ndim!=2:
        raise ValueError('Expected one 2D label plane')
    out=np.zeros(a.shape,dtype=bool)
    h=(a[:,1:]!=a[:,:-1])&(a[:,1:]>0)&(a[:,:-1]>0)
    v=(a[1:,:]!=a[:-1,:])&(a[1:,:]>0)&(a[:-1,:]>0)
    out[:,1:] |= h; out[:,:-1] |= h
    out[1:,:] |= v; out[:-1,:] |= v
    return out


def rasterize_line(points, width, height, inset=2):
    if not isinstance(points,list) or len(points)<2:
        raise ValueError('Polyline needs >= 2 vertices')
    p=np.asarray(points,dtype=float)
    if p.shape != (len(points),2) or not np.isfinite(p).all():
        raise ValueError('Invalid polyline points')
    if np.any(p[:,0]<0) or np.any(p[:,1]<0) or np.any(p[:,0]>=width) or np.any(p[:,1]>=height):
        raise ValueError('Polyline has points outside tile')
    coords=[]
    for a,b in zip(p[:-1],p[1:]):
        n=max(2,int(np.ceil(np.max(np.abs(a-b))*2))+1)
        coords.extend(np.rint(np.linspace(a,b,n)).astype(int).tolist())
    coords=np.unique(np.asarray(coords),axis=0)
    good=(coords[:,0]>=inset)&(coords[:,0]<width-inset)&(coords[:,1]>=inset)&(coords[:,1]<height-inset)
    return coords[good]  # x,y local


def evaluate(args):
    p=Path(args.review_dir)
    manifest=json.loads((p/'annotation_manifest.json').read_text(encoding='utf-8'))
    patch_by_id={x['id']:x for x in manifest['patches']}
    if len(patch_by_id)!=len(manifest['patches']):
        raise RuntimeError('Duplicate patch ID')
    masks={
        'nuclear_voronoi':np.asarray(tifffile.imread(args.voronoi_labels)),
        'cadherin_assisted':np.asarray(tifffile.imread(args.cadherin_labels))
    }
    x0,x1,y0,y1=manifest['roi_xy']
    expected=(16,y1-y0,x1-x0)
    for key,a in masks.items():
        if a.shape!=expected or not np.issubdtype(a.dtype,np.integer):
            raise ValueError(f'{key} expected integer {expected} mask, got {a.shape} {a.dtype}')
    if args.tolerance<0 or args.max_distance<=0:
        raise ValueError('Invalid distance settings')
    # Important: distances computed on full ROI so patch edge never truncates model interfaces
    distances={}
    for name, a in masks.items():
        distances[name]={}
        for z in manifest['z_indices']:
            edge=shared_interface(a[z])
            distances[name][z]=(distance_transform_edt(~edge) if edge.any()
                                else np.full(edge.shape,float(sum(edge.shape)+1)))
    rows=[]
    for f in args.annotations:
        record=json.loads(Path(f).read_text(encoding='utf-8'))
        if record.get('schema_version') != VERSION:
            raise ValueError(f'Annotation schema mismatch: {f}')
        if record.get('reference_channels') != ['Ir193','Cd112','Gd160']:
            raise ValueError(f'Unexpected reference channels in {f}')
        if record.get('patch_set_token') != manifest.get('patch_set_token'):
            raise ValueError(f'Patch set token mismatch in {f}')
        reader=record.get('annotator_id','').strip()
        if not reader:
            raise ValueError(f'Missing annotator ID: {f}')
        for line_id,line in enumerate(record.get('lines',[])):
            pid=line.get('patch_id')
            if pid not in patch_by_id:
                raise ValueError(f'Unknown patch ID: {pid}')
            category=line.get('category')
            if category not in ('visible','uncertain'):
                raise ValueError(f'Invalid category {category}')
            patch=patch_by_id[pid]
            points=rasterize_line(line.get('points_xy_local'),patch['width'],patch['height'])
            if not len(points):
                continue
            z=patch['z']
            xs=points[:,0]+patch['x0']-x0
            ys=points[:,1]+patch['y0']-y0
            for method in masks:
                ds=distances[method][z][ys,xs]
                rows.append(dict(reader=reader,annotation_file=str(f),patch_id=pid,line_index=line_id,
                                 z=z,category=category,method=method,n_reference_pixels=len(points),
                                 tolerance_px=args.tolerance,
                                 fraction_within_tolerance=float((ds<=args.tolerance).mean()),
                                 mean_boundary_distance_px=float(ds.mean()),
                                 median_boundary_distance_px=float(np.median(ds)),
                                 no_model_boundary_nearby=bool((ds>args.max_distance).all())))
    if not rows:
        raise ValueError('No valid lines recorded; annotate first')
    import csv
    out=Path(args.out)
    if out.exists():
        raise FileExistsError(f'No overwrite: {out}')
    out.mkdir(parents=True)
    cols=list(rows[0].keys())
    with (out/'per_line_boundary_proximity.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
    import pandas as pd
    df=pd.DataFrame(rows)
    visible=df[df.category=='visible'].copy()
    summary=dict(status='EXPLORATORY',version=VERSION,
        reference_channels=['Ir193','Cd112','Gd160'],source_annotation_files=list(map(str,args.annotations)),
        label_paths={k:str(v) for k,v in [('nuclear_voronoi',args.voronoi_labels),('cadherin_assisted',args.cadherin_labels)]},
        label_sha256={k:sha256(v) for k,v in [('nuclear_voronoi',args.voronoi_labels),('cadherin_assisted',args.cadherin_labels)]},
        annotators=sorted(df.reader.unique().tolist()),
        n_visible_lines=int(visible[['reader','annotation_file','patch_id','line_index']].drop_duplicates().shape[0]),
        n_uncertain_lines=int((df.category=='uncertain').sum()//2),
        n_patches_with_visible_lines=int(visible.patch_id.nunique()),
        number_of_review_patches=len(manifest['patches']),
        comment='Sparse-line boundary recall/proximity only; not precision/F1/Dice. One biological specimen.')
    if len(visible):
        for method,g in visible.groupby('method'):
            summary[method]={"n_lines":len(g),"mean_of_per_line_recall":float(g.fraction_within_tolerance.mean()),
                  "median_of_per_line_distance_px":float(g.median_boundary_distance_px.median())}
        # matched difference requires identical reader/patch/line, which is true for these rows
        keyed=['reader','annotation_file','patch_id','line_index']
        wide=visible.pivot(index=keyed,columns='method',values='fraction_within_tolerance')
        delta=wide['cadherin_assisted']-wide['nuclear_voronoi']
        summary['paired_difference_cadherin_minus_voronoi']={
            'n_annotated_lines':int(len(delta)),
            'mean_recall_difference':float(delta.mean()),
            'median_recall_difference':float(delta.median()),
            'fraction_lines_cadherin_better':float((delta>0).mean()),
            'fraction_lines_tied':float((delta==0).mean())}
        summary['by_reader']={}
        for reader, subset in visible.groupby('reader'):
            w=subset.pivot(index=['annotation_file','patch_id','line_index'], columns='method', values='fraction_within_tolerance')
            dd=w['cadherin_assisted']-w['nuclear_voronoi']
            summary['by_reader'][reader]={'n_lines':len(dd),'mean_paired_difference':float(dd.mean())}
        print(f'Paired reference-line proximity (tolerance={args.tolerance:g}px):')
        print(json.dumps(summary['paired_difference_cadherin_minus_voronoi'],indent=2))
    (out/'evaluation_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(dict(status='PASS',mode='evaluate',n_visible_lines=summary['n_visible_lines'],
                          n_uncertain_lines=summary['n_uncertain_lines'],out=str(out)),indent=2))


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='mode',required=True)
    a=sub.add_parser('prepare',help='Generate blind reference images and local browser annotation app')
    a.add_argument('--source-zip',type=Path,default=Path('data/inspect/LVILymph/LVILymphBreastCancer_SIMILARITY10_In115.zip'))
    a.add_argument('--roi',type=int,nargs=4,default=[480,720,480,720],metavar=('X0','X1','Y0','Y1'))
    a.add_argument('--z',type=int,nargs='+',default=DEFAULT_Z)
    a.add_argument('--tile',type=int,default=80)
    a.add_argument('--percentile',type=float,default=99.5)
    a.add_argument('--out',type=Path,default=Path('runs/3d_independent_boundary_review_v0_1'))
    b=sub.add_parser('verify',help='Verify locked annotation preparation files')
    b.add_argument('--out',type=Path,default=Path('runs/3d_independent_boundary_review_v0_1'))
    c=sub.add_parser('evaluate',help='Compare frozen Workflow11H boundaries with visible manual polylines')
    c.add_argument('--review-dir',type=Path,default=Path('runs/3d_independent_boundary_review_v0_1'))
    c.add_argument('--annotations',type=Path,nargs='+',required=True)
    c.add_argument('--voronoi-labels',type=Path,default=Path('runs/3d_nucleus_constrained_celllike_segmentation_v0_2/baseline_voronoi_labels_zyx.tif'))
    c.add_argument('--cadherin-labels',type=Path,default=Path('runs/3d_nucleus_constrained_celllike_segmentation_v0_2/candidate_cadherin_labels_zyx.tif'))
    c.add_argument('--tolerance',type=float,default=2.)
    c.add_argument('--max-distance',type=float,default=10.)
    c.add_argument('--out',type=Path,default=Path('runs/3d_independent_boundary_evaluation_v0_1'))
    args=ap.parse_args(argv)
    if args.mode=='prepare':prepare(args)
    elif args.mode=='verify':verify(args)
    else:evaluate(args)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        print('ERROR:',type(exc).__name__,str(exc),file=sys.stderr)
        raise SystemExit(1)
