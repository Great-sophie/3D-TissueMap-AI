"""CLI commands: synth, inspect, convert, train, predict, evaluate, quantify, visualize, demo."""
import argparse
from pathlib import Path
import json
from .data import generate_synthetic, inspect_manifest, convert_sample
from .train import train_model, predict_manifest
from .evaluate import evaluate
from .spatial import quantify
from .visualize import save_overview, save_cell_atlas
from .fusion import fuse


def parser():
    p=argparse.ArgumentParser(prog="tissuemap",description="Reproducible 3D tissue image/spatial protein analysis prototype")
    sub=p.add_subparsers(dest="command",required=True)
    s=sub.add_parser("synth",help="Generate clearly labelled synthetic test dataset")
    s.add_argument("--out",default="data/synthetic");s.add_argument("--samples",type=int,default=6)
    s=sub.add_parser("inspect",help="Validate manifest, shapes, annotations, split leakage")
    s.add_argument("--manifest",required=True);s.add_argument("--out")
    s=sub.add_parser("convert",help="Convert manually inspected TIFF or NPY images and instance masks to canonical NPZ")
    for flag in ("image", "mask", "image-axes", "mask-axes", "channels", "sample-id", "patient-id", "split", "output", "manifest"):
        s.add_argument("--"+flag,required=True)
    s.add_argument("--spacing",nargs=3,type=float,required=True,metavar=("Z","Y","X"))
    s=sub.add_parser("train",help="Train an actual 2D or 3D U-Net")
    s.add_argument("--manifest",required=True);s.add_argument("--model",choices=["2d","3d"],required=True)
    s.add_argument("--output",required=True);s.add_argument("--epochs",type=int,default=2)
    s.add_argument("--channel",default="Nuclear");s.add_argument("--base",type=int,default=8)
    s.add_argument("--device",default="auto",choices=["auto","cpu","cuda"])
    s.add_argument("--patch",nargs=3,type=int,default=(12,48,48))
    s.add_argument("--steps-per-sample",type=int,default=2)
    s=sub.add_parser("predict",help="Run held-out inference; 3D uses sliding windows")
    s.add_argument("--manifest",required=True);s.add_argument("--checkpoint",required=True)
    s.add_argument("--output",required=True);s.add_argument("--split",default="test",choices=["train","val","test","all"])
    s.add_argument("--patch",nargs=3,type=int,default=(12,48,48))
    s.add_argument("--device",default="auto",choices=["auto","cpu","cuda"])
    s=sub.add_parser("evaluate")
    s.add_argument("--manifest",required=True);s.add_argument("--predictions",required=True)
    s.add_argument("--output",required=True);s.add_argument("--split",default="test")
    s=sub.add_parser("quantify")
    s.add_argument("--manifest",required=True);s.add_argument("--output",required=True)
    s.add_argument("--split",default="test");s.add_argument("--predictions")
    s.add_argument("--predicted-components",action="store_true",help="Use connected components of predicted semantic mask instead of reference cell instances")
    s=sub.add_parser("fuse",help="Exploratory fusion: morphology + spatial protein cell features")
    s.add_argument("--cells",required=True);s.add_argument("--output",required=True)
    s.add_argument("--clusters",type=int,default=2)
    s=sub.add_parser("visualize")
    s.add_argument("--manifest",required=True);s.add_argument("--output",required=True)
    s.add_argument("--sample-id");s.add_argument("--predictions")
    s=sub.add_parser("demo",help="Run full pipeline on SYNTHETIC volumes (smoke test only)")
    s.add_argument("--output",default="runs/demo");s.add_argument("--epochs",type=int,default=1)
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    cmd=args.command
    if cmd=="synth":
        print(f"Synthetic manifest: {generate_synthetic(args.out,args.samples)}")
    elif cmd=="inspect":
        print(json.dumps(inspect_manifest(args.manifest,args.out),indent=2))
    elif cmd=="convert":
        print(convert_sample(args.image,args.mask,args.image_axes,args.mask_axes,
                             [x.strip() for x in args.channels.split(',')], args.sample_id,args.patient_id,
                             args.split,args.spacing,args.output,args.manifest))
    elif cmd=="train":
        ck,h=train_model(args.manifest,3 if args.model=="3d" else 2,args.output,args.channel,args.epochs,
                         base=args.base,device=args.device,patch_shape=args.patch,steps_per_sample=args.steps_per_sample)
        print(json.dumps({"checkpoint":str(ck),"history":h},indent=2))
    elif cmd=="predict":
        print(predict_manifest(args.manifest,args.checkpoint,args.output,args.split,patch=args.patch,device=args.device))
    elif cmd=="evaluate":
        print(json.dumps(evaluate(args.manifest,args.predictions,args.output,args.split),indent=2))
    elif cmd=="quantify":
        print(json.dumps(quantify(args.manifest,args.output,args.split,args.predictions,
                                  not args.predicted_components),indent=2))
    elif cmd=="fuse":
        print(json.dumps(fuse(args.cells,args.output,k=args.clusters),indent=2))
    elif cmd=="visualize":
        print(save_overview(args.manifest,args.output,args.sample_id,args.predictions))
    elif cmd=="demo":
        root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=True)
        manifest=generate_synthetic(root/"data",samples=6)
        inspect_manifest(manifest,root/"qc.json")
        result={"data_kind":"SYNTHETIC ONLY; not real breast cancer data", "manifest":str(manifest)}
        for dim in (2,3):
            work=root/f"unet_{dim}d"
            ck,history=train_model(manifest,dim,work,epochs=args.epochs,base=4,device="cpu")
            predict_manifest(manifest,ck,work/"predictions",patch=(12,48,48),device="cpu")
            result[f"unet_{dim}d"] = evaluate(manifest,work/"predictions",work/"evaluation")
            save_overview(manifest,work/"preview.png",predictions=work/"predictions",
                          sample_id="synthetic_05")
        result["molecular_quantification"] = quantify(manifest,root/"spatial",split="test")
        result["morphology_protein_fusion"] = fuse(root/"spatial"/"cells.csv",root/"fusion")
        save_cell_atlas(root/"spatial"/"cells.csv",root/"spatial"/"3d_cell_atlas.png")
        (root/"demo_summary.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
        print(json.dumps(result,indent=2))

if __name__ == "__main__":
    main()
