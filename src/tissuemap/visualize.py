"""Simple, transparent qualitative QC visualizations."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .data import read_manifest, load_sample, channel_image, normalize_image


def save_overview(manifest, output, sample_id=None, predictions=None, channel="Nuclear"):
    df=read_manifest(manifest)
    rows=df[df.sample_id==sample_id] if sample_id else df.head(1)
    if len(rows)!=1:
        raise ValueError("Select one valid sample ID")
    row=next(rows.itertuples())
    image,mask,names=load_sample(row.path)
    x=normalize_image(channel_image(image,names,channel))
    z=x.shape[0]//2
    plots=[(x[z],f"{channel}: XY at Z={z}"),(mask[z]>0,"Reference foreground")]
    if predictions:
        with np.load(Path(predictions)/f"{row.sample_id}.npz",allow_pickle=False) as f:
            pred=f["semantic_mask"]
        plots.append((pred[z],"Predicted foreground"))
    fig,axs=plt.subplots(1,len(plots),figsize=(4*len(plots),4))
    for ax,(panel,title) in zip(np.atleast_1d(axs),plots):
        ax.imshow(panel,cmap="gray",interpolation="nearest");ax.set_title(title);ax.axis("off")
    fig.suptitle(f"{row.sample_id}: qualitative QC (not ground-truth audit)")
    fig.tight_layout()
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(out,dpi=140);plt.close(fig)
    return out


def save_cell_atlas(cells_csv, output):
    """Interactive-research-ready *static* 3D centroid plot, physical units, candidate gates."""
    import pandas as pd
    cells=pd.read_csv(cells_csv)
    needed={"centroid_x_um","centroid_y_um","centroid_z_um"}
    if not needed.issubset(cells.columns):
        raise ValueError(f"Missing centroid coordinates: {needed-set(cells.columns)}")
    fig=plt.figure(figsize=(7,6))
    ax=fig.add_subplot(111,projection="3d")
    if "CD8_high_exploratory" in cells:
        groups=[("candidate CD8-high",cells["CD8_high_exploratory"]==True),
                ("other",cells["CD8_high_exploratory"]!=True)]
    else:
        groups=[("all cells",np.ones(len(cells),dtype=bool))]
    for name,group in groups:
        subset=cells[group]
        ax.scatter(subset.centroid_x_um,subset.centroid_y_um,subset.centroid_z_um,
                   s=15+0.5*np.sqrt(subset.volume_um3) if "volume_um3" in subset else 25,
                   alpha=.75,label=name)
    ax.set(xlabel="X (µm)",ylabel="Y (µm)",zlabel="Z (µm)")
    ax.set_title("Exploratory 3D cell centroids; marker-high is not a validated cell type")
    ax.legend(loc="upper right")
    fig.tight_layout()
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=160);plt.close(fig)
    return output
