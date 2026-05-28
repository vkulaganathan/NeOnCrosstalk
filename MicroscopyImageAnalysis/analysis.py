"""
=============================================================================
IF Analysis  —  HEK293 + SH-SY5Y  |  MYC, pSTAT3, pERK1/2
=============================================================================
Runs all 4 experiments in one go. Output in one folder per experiment.

CHANNEL ASSIGNMENT
------------------
HEK293:
    Red  (Cy3 / Alexa 594, index 0) = pSTAT3
    Green (GFP / Alexa 488, index 1) = pERK1/2

SH-SY5Y:  ← swapped secondaries!
    Red  (Cy3 / Alexa 594, index 0) = pERK1/2
    Green (GFP / Alexa 488, index 1) = pSTAT3

MYC experiments: Cy3 only (red channel)

OUTPUT
------
    HEK293_pSTAT3_pERK/
        plots/        GFP_pERK_nuclear.png+svg  Cy3_pSTAT3_nuclear.png+svg  ...
        statistics/   GFP_pERK_stats.txt  ...
        data/         nuclear_per_cell.csv  overall_per_cell.csv

    HEK293_MYC/
        plots/        Cy3_MYC_nuclear.png+svg  ...
        statistics/   ...
        data/         ...

    SHSY5Y_pSTAT3_pERK/
        ...

    SHSY5Y_MYC/
        ...

Run
---
    python analysis.py
=============================================================================
"""

import csv, os, glob, warnings
from pathlib import Path
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy import ndimage
from scipy.stats import mannwhitneyu, kruskal, ttest_ind
from skimage import filters, morphology, segmentation, measure, exposure
from skimage.feature import peak_local_max
from skimage.segmentation import find_boundaries

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

EXPERIMENTS = [

    {
        "name":       "HEK293_pSTAT3_pERK",
        "folders":    {"0 µM": "RM_0uM", "0.5 µM": "RM_0.5uM", "1 µM": "RM_1uM"},
        "channel_red":   "pSTAT3",   # Alexa 594 → index 0
        "channel_green": "pERK1/2",  # Alexa 488 → index 1
        "has_green": True,
    },

    {
        "name":       "HEK293_MYC",
        "folders":    {"0 µM": "RM_0uM", "0.5 µM": "RM_0.5uM", "1 µM": "RM_1uM"},
        "channel_red":   "MYC",      # Alexa 594 → index 0
        "channel_green": None,       # no green marker
        "has_green": False,
    },

    {
        "name":       "SHSY5Y_pSTAT3_pERK",
        "folders":    {"0 µM": "SH_0uM", "5 µM": "SH_5uM", "10 µM": "SH_10uM"},
        "channel_red":   "pERK1/2",  # Alexa 594 → index 0  ← swapped!
        "channel_green": "pSTAT3",   # Alexa 488 → index 1  ← swapped!
        "has_green": True,
    },

    {
        "name":       "SHSY5Y_MYC",
        "folders":    {"0 µM": "SH_0uM", "5 µM": "SH_5uM", "10 µM": "SH_10uM"},
        "channel_red":   "MYC",      # Alexa 594 → index 0
        "channel_green": None,
        "has_green": False,
    },

]

# ══════════════════════════════════════════════════════════════════════════════
# SHARED SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

MIN_CELL_AREA         = 1_000
DAPI_SIGMA            = 3
NUCLEUS_SEP_FP        = 25
NUCLEUS_EROSION       = 2
CYTOPLASM_DILATION_PX = 20

SCALEBAR_AUTO_DETECT  = True
SCALEBAR_CORNER_FRAC  = 0.20
SCALEBAR_BRIGHT_PCT   = 92
SCALEBAR_MIN_WIDTH    = 40
SCALEBAR_MAX_HEIGHT   = 30
SCALEBAR_BUFFER       = 8

OUTLIER_IQR_FACTOR = 1.5
FDR_ALPHA          = 0.05
FIGURE_DPI         = 150

BG      = "white";    FG      = "#111111"
AX_BG   = "#f7f7f7";  GRID_C  = "#dddddd";  SPINE_C = "#aaaaaa"
OUTLIER_COLOR = "#e31a1c"

CH_DAPI  = 2
CH_RED   = 0   # Alexa 594 / Cy3
CH_GREEN = 1   # Alexa 488 / GFP

plt.rcParams.update({
    "figure.facecolor": BG,    "axes.facecolor":  AX_BG,
    "axes.edgecolor":   SPINE_C, "axes.labelcolor": FG,
    "xtick.color": FG, "ytick.color": FG, "text.color": FG,
    "grid.color":  GRID_C,
    "legend.facecolor": "white", "legend.edgecolor": SPINE_C,
})

rng = np.random.default_rng(42)


# ══════════════════════════════════════════════════════════════════════════════
# STATISTICS HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def tukey(a):
    q1, q3 = np.percentile(a, [25, 75]); iqr = q3 - q1
    return (a < q1 - OUTLIER_IQR_FACTOR*iqr) | (a > q3 + OUTLIER_IQR_FACTOR*iqr)

def sig(p):
    if p is None or (isinstance(p, float) and np.isnan(p)): return "n/a"
    return "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else "ns"))

def pf(p):
    if p is None or (isinstance(p, float) and np.isnan(p)): return "n/a"
    return "<0.001" if p<0.001 else f"{p:.5f}"

def bh(p_values):
    n = len(p_values)
    if n == 0: return []
    idx = sorted(range(n), key=lambda i: p_values[i])
    q = [1.0]*n; prev = 1.0
    for rank, i in enumerate(reversed(idx), 1):
        prev = min(p_values[i]*n/(n-rank+1), prev); q[i] = min(prev, 1.0)
    return q

def run_stats(arrs, labels):
    pairs = list(combinations(labels, 2))
    tests = []
    try:    kH, kP = kruskal(*[arrs[l] for l in labels])
    except: kH, kP = None, None
    for a, b in pairs:
        va, vb = arrs[a], arrs[b]
        try:    _, p_mwu = mannwhitneyu(va, vb, alternative="two-sided")
        except: p_mwu = None
        try:    _, p_t   = ttest_ind(va, vb, equal_var=False)
        except: p_t = None
        tests.append({"A":a,"B":b,"n_A":len(va),"n_B":len(vb),
                      "p_mwu":p_mwu,"p_t":p_t})
    # BH
    for pk in ["p_mwu","p_t"]:
        valid = [(i,t) for i,t in enumerate(tests) if t[pk] is not None]
        if valid:
            qs = bh([t[pk] for _,t in valid])
            for (i,_),q in zip(valid,qs): tests[i]["q_"+pk[2:]] = round(q,6)
    return {"tests":tests,"kH":kH,"kP":kP}


# ══════════════════════════════════════════════════════════════════════════════
# SCALE BAR
# ══════════════════════════════════════════════════════════════════════════════

def scalebar_mask(img):
    h, w = img.shape[:2]
    mask = np.zeros((h,w), dtype=bool)
    if not SCALEBAR_AUTO_DETECT: return mask
    ch, cw = int(h*SCALEBAR_CORNER_FRAC), int(w*SCALEBAR_CORNER_FRAC)
    corner = np.zeros((h,w), dtype=bool)
    corner[:ch,:cw]=corner[:ch,w-cw:]=corner[h-ch:,:cw]=corner[h-ch:,w-cw:]=True
    bright = img.max(axis=2).astype(np.float32)
    cv = bright[corner]
    if not cv.size: return mask
    t = np.percentile(cv, SCALEBAR_BRIGHT_PCT)
    c = ndimage.binary_opening((bright>t)&corner, structure=np.ones((2,5)))
    labeled, n = ndimage.label(c)
    for rid in range(1,n+1):
        reg = labeled==rid; rows,cols = np.where(reg)
        if not rows.size: continue
        rh=rows.max()-rows.min()+1; rw=cols.max()-cols.min()+1
        if rw>=SCALEBAR_MIN_WIDTH and rh<=SCALEBAR_MAX_HEIGHT and reg.sum()>=0.3*rw*rh:
            mask[max(0,rows.min()-SCALEBAR_BUFFER):min(h,rows.max()+SCALEBAR_BUFFER+1),
                 max(0,cols.min()-SCALEBAR_BUFFER):min(w,cols.max()+SCALEBAR_BUFFER+1)]=True
    return mask


# ══════════════════════════════════════════════════════════════════════════════
# SEGMENTATION + MEASUREMENT
# ══════════════════════════════════════════════════════════════════════════════

def process_image(img_path, condition, has_green, save_mask_dir=None):
    img = tifffile.imread(str(img_path))
    if img.ndim==3 and img.shape[0]==3: img = np.moveaxis(img,0,-1)

    sb  = scalebar_mask(img)
    img = img.copy().astype(np.float32)
    if sb.any(): img[sb] = 0

    dapi = img[:,:,CH_DAPI]
    red  = img[:,:,CH_RED]
    grn  = img[:,:,CH_GREEN] if has_green else None

    # segmentation
    smooth = filters.gaussian(dapi, sigma=DAPI_SIGMA)
    thresh = filters.threshold_otsu(smooth)
    binary = ndimage.binary_fill_holes(smooth>thresh)
    binary = morphology.remove_small_objects(binary, min_size=MIN_CELL_AREA)
    if sb.any(): binary[sb]=False
    binary = morphology.binary_erosion(binary, morphology.disk(NUCLEUS_EROSION))
    dist   = ndimage.distance_transform_edt(binary)
    coords = peak_local_max(dist,
                            footprint=np.ones((NUCLEUS_SEP_FP,NUCLEUS_SEP_FP)),
                            labels=binary)
    mk=np.zeros(dist.shape,dtype=bool); mk[tuple(coords.T)]=True
    markers,_  = ndimage.label(mk)
    nuc_labels = segmentation.watershed(-dist, markers, mask=binary)

    # vectorized cell dilation
    bg_mask         = nuc_labels==0
    dist2,idx       = ndimage.distance_transform_edt(bg_mask, return_indices=True)
    cell_labels     = np.where(bg_mask&(dist2<=CYTOPLASM_DILATION_PX),
                               nuc_labels[idx[0],idx[1]], nuc_labels)

    # background + clip
    bg_reg = (cell_labels==0)&(~sb)
    bg_r   = float(np.median(red[bg_reg]))
    rc     = np.clip(red-bg_r, 0, None)
    gc     = None
    if has_green:
        bg_g = float(np.median(grn[bg_reg]))
        gc   = np.clip(grn-bg_g, 0, None)

    # measure
    nuc_r  = measure.regionprops(nuc_labels,  intensity_image=rc)
    cell_r = {p.label:p for p in measure.regionprops(cell_labels, intensity_image=rc)}
    nuc_g  = measure.regionprops(nuc_labels,  intensity_image=gc) if has_green else []
    cell_g = {p.label:p for p in measure.regionprops(cell_labels, intensity_image=gc)} \
             if has_green else {}

    # save mask for Methods
    if save_mask_dir:
        _save_mask(img, sb, nuc_labels, cell_labels,
                   [p for p in nuc_r if p.area>=MIN_CELL_AREA
                    and not sb[int(p.centroid[0]),int(p.centroid[1])]],
                   thresh, img_path, condition, save_mask_dir)

    cells = []
    for i,(p_r, p_g) in enumerate(zip(nuc_r,
                                       nuc_g if nuc_g else [None]*len(nuc_r))):
        if p_r.area < MIN_CELL_AREA: continue
        cy,cx = int(p_r.centroid[0]),int(p_r.centroid[1])
        if sb[cy,cx]: continue
        cr = cell_r.get(p_r.label)
        cg = cell_g.get(p_r.label) if has_green else None
        nr = float(p_r.mean_intensity)
        ng = float(p_g.mean_intensity) if p_g else None
        ovr= float(cr.mean_intensity)  if cr else nr
        ovg= float(cg.mean_intensity)  if cg else ng
        cells.append({
            "condition":  condition,
            "image":      Path(img_path).name,
            "area_nuc":   int(p_r.area),
            "area_cell":  int(cr.area) if cr else int(p_r.area),
            "nuc_red":    nr,
            "nuc_green":  ng,
            "cell_red":   ovr,
            "cell_green": ovg,
            "nc_red":     nr/ovr if ovr>0.001 else None,
            "nc_green":   ng/ovg if (ovg and ovg>0.001) else None,
        })
    return cells


def _run(args): return process_image(*args)


# ══════════════════════════════════════════════════════════════════════════════
# MASK IMAGE FOR METHODS
# ══════════════════════════════════════════════════════════════════════════════

def _save_mask(img, sb, nuc_labels, cell_labels, accepted,
               thresh, img_path, condition, out_dir):
    rgb = np.stack([
        exposure.rescale_intensity(img[:,:,0], out_range=(0,255)),
        exposure.rescale_intensity(img[:,:,1], out_range=(0,255)),
        exposure.rescale_intensity(img[:,:,2], out_range=(0,255)),
    ], axis=-1).astype(np.uint8)

    b_nuc = find_boundaries(nuc_labels, mode="outer")
    acc_ids = [p.label for p in accepted]
    b_acc = find_boundaries(
        np.where(np.isin(nuc_labels,acc_ids),nuc_labels,0), mode="outer")
    final = rgb.copy(); final[b_acc]=[255,255,0]

    fig,axes = plt.subplots(1,3,figsize=(18,6),facecolor="white")
    fig.suptitle(f"{condition}  |  {Path(img_path).name}  |  "
                 f"{len(accepted)} cells accepted",
                 fontsize=10,fontweight="bold",y=1.01)

    ax=axes[0]; ax.imshow(rgb); ax.axis("off")
    ax.set_title("Raw composite\nRed=Cy3/594  Green=GFP/488  Blue=DAPI",
                 fontsize=9,fontweight="bold")

    ax=axes[1]
    ax.imshow(exposure.rescale_intensity(img[:,:,CH_DAPI],out_range=(0,1)),cmap="Blues")
    ov=np.zeros((*img.shape[:2],4)); ov[b_nuc]=[1,1,0,1]
    ax.imshow(ov); ax.axis("off")
    ax.set_title(f"Nucleus mask\nOtsu={thresh:.0f}  +  watershed\n"
                 f"{len(accepted)} nuclei (yellow)",
                 fontsize=9,fontweight="bold")

    ax=axes[2]; ax.imshow(final)
    for i,p in enumerate(accepted,1):
        cy,cx=p.centroid
        ax.text(cx,cy,str(i),color="yellow",fontsize=6,
                ha="center",va="center",fontweight="bold")
    ax.axis("off")
    ax.set_title(f"Accepted cells ({len(accepted)})\n"
                 f"Yellow = included  |  Cell = nucleus+{CYTOPLASM_DILATION_PX}px",
                 fontsize=9,fontweight="bold")

    plt.tight_layout()
    safe=condition.replace(" ","_").replace("/","_").replace("µ","u")
    out =Path(out_dir)/f"{safe}_mask.png"
    fig.savefig(out,dpi=FIGURE_DPI,bbox_inches="tight",facecolor="white")
    fig.savefig(str(out).replace(".png",".svg"),bbox_inches="tight",facecolor="white")
    plt.close(fig)
    print(f"      mask → {out.name} +svg")


# ══════════════════════════════════════════════════════════════════════════════
# BAR PLOT
# ══════════════════════════════════════════════════════════════════════════════

def make_bar(arrs, labels, marker_name, compartment, ylabel,
             stats, ctrl_label, out_dir, filename):
    means = [np.mean(arrs[l]) for l in labels]
    sems  = [np.std(arrs[l])/np.sqrt(len(arrs[l])) for l in labels]

    # colours — evenly spaced blues→reds
    cmap = plt.cm.get_cmap("RdYlBu_r", len(labels))
    cols = [cmap(i/(max(len(labels)-1,1))) for i in range(len(labels))]
    # override if palette matches
    PALETTE={"0 µM":"#2166ac","0.5 µM":"#f4a582","1 µM":"#b2182b",
             "5 µM":"#f4a582","10 µM":"#b2182b"}
    cols=[PALETTE.get(l,cols[i]) for i,l in enumerate(labels)]

    fig,ax=plt.subplots(figsize=(8,6.5),facecolor=BG)
    ax.set_facecolor(AX_BG)
    for sp in ax.spines.values(): sp.set_color(SPINE_C)

    x=np.arange(len(labels))
    ax.bar(x,means,width=0.5,color=cols,edgecolor="white",lw=0.9,zorder=3,alpha=0.85)
    ax.errorbar(x,means,yerr=sems,fmt="none",color="#333",
                capsize=8,capthick=2,elinewidth=2,zorder=5)

    for i,lbl in enumerate(labels):
        v=arrs[lbl]; out=tukey(v)
        jx=rng.uniform(-0.18,0.18,len(v))
        ax.scatter(i+jx[~out],v[~out],s=20,alpha=0.50,
                   color=cols[i],edgecolors="none",zorder=4)
        if out.any():
            ax.scatter(i+jx[out],v[out],s=60,marker="D",
                       facecolors="none",edgecolors=OUTLIER_COLOR,
                       linewidths=1.3,zorder=6,
                       label="outlier" if i==0 else "")
        ax.text(i,0,f"n={len(v)}",ha="center",va="bottom",
                color="white",fontsize=8,fontweight="bold")

    for i,(m,s) in enumerate(zip(means,sems)):
        ax.text(i,m+s+abs(max(means))*0.03,f"{m:.2f}×",
                ha="center",va="bottom",color=FG,fontsize=11,fontweight="bold")

    ax.axhline(1.0,color="#666",lw=1.2,ls="--",zorder=2,label="control (1×)")

    # brackets ctrl vs each
    y_top=max(m+s for m,s in zip(means,sems))
    step =abs(max(means))*0.18+0.30
    ctrl_idx=labels.index(ctrl_label)
    for k,lbl in enumerate([l for l in labels if l!=ctrl_label]):
        t=next((t for t in stats["tests"]
                if t["A"]==ctrl_label and t["B"]==lbl),None)
        q=t.get("q_mwu") if t else None
        ib=labels.index(lbl)
        col_bkt="#2166ac" if (q is not None and q<FDR_ALPHA) else "#aaaaaa"
        y=y_top+abs(max(means))*0.12+k*step
        ax.plot([ctrl_idx,ctrl_idx,ib,ib],
                [y,y+step*0.22,y+step*0.22,y],
                color=col_bkt,lw=1.0,zorder=8)
        ax.text((ctrl_idx+ib)/2,y+step*0.24,
                f"q={pf(q)}\n{sig(q)}",
                ha="center",va="bottom",color=col_bkt,fontsize=8,fontweight="bold")

    ax.set_ylim(0,y_top+abs(max(means))*0.12+(len(labels)-1)*step+step*0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels,color=FG,fontsize=11,fontweight="bold")
    ax.tick_params(colors=FG,labelsize=10)
    ax.set_ylabel(ylabel,color=FG,fontsize=11)
    ax.set_title(f"{marker_name}  —  {compartment}\n"
                 f"mean ± SEM  ·  dots=cells  ·  ◇=outlier  ·  q=BH-FDR",
                 color=FG,fontsize=11,fontweight="bold",pad=6)
    ax.grid(axis="y",color=GRID_C,linewidth=0.7,zorder=0)
    handles,_ =ax.get_legend_handles_labels()
    if handles: ax.legend(fontsize=8,loc="upper left")

    plt.tight_layout()
    out=Path(out_dir)/f"{filename}.png"
    fig.savefig(out,dpi=FIGURE_DPI,bbox_inches="tight",facecolor=BG)
    fig.savefig(str(out).replace(".png",".svg"),bbox_inches="tight",facecolor=BG)
    plt.close(fig)
    print(f"      {out.name} +svg")


# ══════════════════════════════════════════════════════════════════════════════
# STATISTICS TXT
# ══════════════════════════════════════════════════════════════════════════════

def save_stats_txt(arrs, labels, marker_name, compartment, stats, out_dir, filename):
    lines=[
        "="*90,
        f"  {marker_name}  —  {compartment}",
        f"  Mann-Whitney U + Benjamini-Hochberg FDR (α={FDR_ALPHA})",
        "="*90,"",
        f"  {'Condition':12}  {'n':>5}  {'Mean×':>8}  {'SD':>8}  "
        f"{'SEM':>8}  {'Median×':>8}  {'Outliers':>9}",
        "  "+"-"*65,
    ]
    for lbl in labels:
        v=arrs[lbl]
        lines.append(f"  {lbl:12}  {len(v):>5}  {np.mean(v):>8.4f}  "
                     f"{np.std(v):>8.4f}  {np.std(v)/np.sqrt(len(v)):>8.4f}  "
                     f"{np.median(v):>8.4f}  {tukey(v).sum():>9}")
    kH,kP=stats["kH"],stats["kP"]
    lines+=["",
            f"  Kruskal-Wallis: H={f'{kH:.3f}' if kH else 'n/a'}  "
            f"p={pf(kP)}  {sig(kP)}","",
            f"  {'Comparison':22}  {'MWU p':>10}  {'MWU q (BH)':>12}  "
            f"{'sig':>5}  {'t-test p':>10}  {'t-q (BH)':>10}  {'sig':>5}",
            "  "+"-"*78]
    for t in stats["tests"]:
        lines.append(
            f"  {t['A']+' vs '+t['B']:22}  "
            f"{pf(t.get('p_mwu')):>10}  {pf(t.get('q_mwu')):>12}  "
            f"{sig(t.get('q_mwu')):>5}  "
            f"{pf(t.get('p_t')):>10}  {pf(t.get('q_t')):>10}  "
            f"{sig(t.get('q_t')):>5}")
    lines.append("="*90)
    with open(Path(out_dir)/f"{filename}.txt","w") as f:
        f.write("\n".join(lines))
    print(f"      {filename}.txt")


# ══════════════════════════════════════════════════════════════════════════════
# RUN ONE EXPERIMENT
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment(exp):
    name       = exp["name"]
    folders    = exp["folders"]
    ch_red     = exp["channel_red"]
    ch_green   = exp["channel_green"]
    has_green  = exp["has_green"]
    labels     = list(folders.keys())
    ctrl_label = labels[0]

    # output folders
    out     = Path(name)
    dir_pl  = out/"plots"
    dir_st  = out/"statistics"
    dir_da  = out/"data"
    dir_ma  = out/"masks"
    for d in [dir_pl,dir_st,dir_da,dir_ma]: d.mkdir(parents=True,exist_ok=True)

    print(f"\n{'═'*60}")
    print(f"  {name}")
    print(f"  Red  (index 0): {ch_red}")
    print(f"  Green (index 1): {ch_green if has_green else '— not used —'}")
    print("═"*60)

    # check folders exist
    missing=[f for f in folders.values() if not Path(f).exists()]
    if missing:
        print(f"  ✗  Missing folders: {missing}  — skipping")
        return

    # load images in parallel
    raw={lbl:[] for lbl in labels}
    for lbl,folder in folders.items():
        tifs=sorted(glob.glob(str(Path(folder)/"*.TIF"))+
                    glob.glob(str(Path(folder)/"*.tif"))+
                    glob.glob(str(Path(folder)/"*.tiff")))
        if not tifs:
            print(f"  ⚠  No TIFs in {folder}"); continue
        print(f"\n  ▶  {lbl}  ({len(tifs)} image(s))")
        tasks=[(p,lbl,has_green,str(dir_ma) if i==0 else None)
               for i,p in enumerate(tifs)]
        if len(tasks)>1:
            results=[None]*len(tasks)
            with ThreadPoolExecutor(max_workers=4) as pool:
                fmap={pool.submit(_run,t):i for i,t in enumerate(tasks)}
                for f in as_completed(fmap): results[fmap[f]]=f.result()
        else:
            results=[_run(t) for t in tasks]
        raw[lbl]=[c for r in results for c in r]
        print(f"     {len(raw[lbl])} cells")

    if not any(raw.values()): return

    # control means
    def ga(lbl,k): return np.array([c[k] for c in raw[lbl] if c.get(k) is not None])
    ctrl_nuc_r  = np.mean(ga(ctrl_label,"nuc_red"))
    ctrl_cell_r = np.mean(ga(ctrl_label,"cell_red"))
    ctrl_nuc_g  = np.mean(ga(ctrl_label,"nuc_green"))  if has_green else 1
    ctrl_cell_g = np.mean(ga(ctrl_label,"cell_green")) if has_green else 1

    # normalised arrays
    def norm(lbl,key,ctrl_val):
        v=ga(lbl,key)
        return v/ctrl_val if ctrl_val>0 else v

    channels=[
        ("nuc_red",   ctrl_nuc_r,  ch_red,   "Nuclear MFI",
         f"{ch_red} nuclear MFI (0 µM=1×)"),
        ("cell_red",  ctrl_cell_r, ch_red,   "Overall cell MFI",
         f"{ch_red} overall MFI (0 µM=1×)"),
    ]
    if has_green:
        channels+=[
            ("nuc_green",  ctrl_nuc_g,  ch_green, "Nuclear MFI",
             f"{ch_green} nuclear MFI (0 µM=1×)"),
            ("cell_green", ctrl_cell_g, ch_green, "Overall cell MFI",
             f"{ch_green} overall MFI (0 µM=1×)"),
        ]

    # BH across all tests together
    all_arrs_for_bh=[]
    all_pairs=list(combinations(labels,2))
    for key,ctrl_val,_,_,_ in channels:
        arrs={l:norm(l,key,ctrl_val) for l in labels}
        for a,b in all_pairs:
            va,vb=arrs[a],arrs[b]
            if len(va)>=2 and len(vb)>=2:
                try:
                    _,p=mannwhitneyu(va,vb,alternative="two-sided")
                    all_arrs_for_bh.append(p)
                except: all_arrs_for_bh.append(1.0)

    print(f"\n  Generating plots + statistics …")

    for key,ctrl_val,marker,compartment,ylabel in channels:
        arrs  = {l:norm(l,key,ctrl_val) for l in labels}
        stats = run_stats(arrs, labels)
        safe  = f"{marker.replace('/','_').replace(' ','_')}_{compartment.replace(' ','_')}"

        make_bar(arrs,labels,marker,compartment,ylabel,
                 stats,ctrl_label,dir_pl,safe)
        save_stats_txt(arrs,labels,marker,compartment,
                       stats,dir_st,safe)

    # export CSVs
    for csv_key,ctrl_r,ctrl_g,fname in [
        ("nuc",   ctrl_nuc_r,  ctrl_nuc_g,  "nuclear_per_cell.csv"),
        ("cell",  ctrl_cell_r, ctrl_cell_g, "overall_per_cell.csv"),
    ]:
        rows=[]
        for lbl in labels:
            for i,c in enumerate(raw[lbl],1):
                row={
                    "condition":lbl,"cell_num":i,"image":c["image"],
                    "area_nuc":c["area_nuc"],"area_cell":c["area_cell"],
                    f"{ch_red}_raw":    round(c[f"{csv_key}_red"],4),
                    f"{ch_red}_fold":   round(c[f"{csv_key}_red"]/
                                             (ctrl_nuc_r if csv_key=="nuc"
                                              else ctrl_cell_r),4),
                }
                if has_green and c.get(f"{csv_key}_green") is not None:
                    row[f"{ch_green}_raw"]  = round(c[f"{csv_key}_green"],4)
                    row[f"{ch_green}_fold"] = round(c[f"{csv_key}_green"]/
                                                    (ctrl_nuc_g if csv_key=="nuc"
                                                     else ctrl_cell_g),4)
                rows.append(row)
        if rows:
            with open(dir_da/fname,"w",newline="") as f:
                w=csv.DictWriter(f,fieldnames=rows[0].keys())
                w.writeheader(); w.writerows(rows)
            print(f"      {fname}  ({len(rows)} cells)")

    print(f"\n  ✓  {name} done")
    print(f"     {name}/")
    print(f"       plots/       — PNG + SVG per measurement")
    print(f"       statistics/  — txt per measurement")
    print(f"       data/        — nuclear_per_cell.csv  overall_per_cell.csv")
    print(f"       masks/       — segmentation QC per condition")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "█"*60)
print("  IF Analysis  —  HEK293 + SH-SY5Y")
print("█"*60)

for exp in EXPERIMENTS:
    run_experiment(exp)

print("\n" + "█"*60)
print("  ALL DONE")
print("█"*60)
print("\n  Output folders:")
for exp in EXPERIMENTS:
    print(f"    {exp['name']}/")
