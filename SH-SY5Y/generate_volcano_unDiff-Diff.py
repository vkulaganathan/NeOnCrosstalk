import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

file_path = "Undiff-vs-Diff_FPKMS.tsv"
df = pd.read_csv(file_path, sep="\t")
df.columns = [c.strip() for c in df.columns]

# 1. Dynamically find columns
gene_col = "gene" if "gene" in df.columns else "gene_id"
logfc_col = "log2(fold_change)"
p_col = "q_value" if "q_value" in df.columns else "p_value"

# 2. Force numeric conversion
df[logfc_col] = pd.to_numeric(df[logfc_col], errors="coerce")
df[p_col] = pd.to_numeric(df[p_col], errors="coerce")

# 3. Clean up INFINITIES (including massive pseudo-inf numbers like 1.79e+308)
max_float_threshold = 1e300
df.loc[df[logfc_col].abs() >= max_float_threshold, logfc_col] = np.nan

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=[logfc_col, p_col])

# 4. Filter out zeros, negative p-values, AND uninformative q-value=1.0 artifacts
df = df[
    (df[p_col] > 0)
    & (df[p_col] < 1.0)
    & np.isfinite(df[logfc_col])
    & np.isfinite(df[p_col])
]

# Invert sign to align plot to "Differentiated vs Undifferentiated" orientation
df[logfc_col] = -df[logfc_col]

# 5. Calculate -log10 p-value
df["neglog10_p"] = -np.log10(df[p_col])

# 6. Classification thresholds
fc_thresh = 1.0
p_thresh = 0.05
y_thresh_val = -np.log10(p_thresh)  # approx 1.30

# 7. Highlight Specific Genes
#genes_to_label = ["RAF1", "MYC", "SYT2", "SYT13", "SYNPR", "SCRG1", "NTRK2", "MAPK3"]
genes_to_label = ["VDAC1", "VDAC2", "SDHA", "SDHB", "IDH1", "IDH2", "CYCS", "MYC", "RAF1"]


df["gene_str"] = df[gene_col].astype(str)
df["gene_id_str"] = df["gene_id"].astype(str)

highlight = df[
    df["gene_str"].isin(genes_to_label) | df["gene_id_str"].isin(genes_to_label)
].copy()

# 8. Modern Seaborn Theme Application
sns.set_theme(style="whitegrid", context="talk")
fig, ax = plt.subplots(figsize=(12, 9))

# --- NEW: Shading the Significant Quadrants ---
# Left Side Upper Quadrant: Higher in Undifferentiated (Light Green)
ax.axvspan(
    xmin=-15,
    xmax=-fc_thresh,
    ymin=y_thresh_val,
    ymax=14,
    color="#e2f0d9",
    alpha=0.5,
    zorder=0,
)
# Right Side Upper Quadrant: Higher in Differentiated (Light Red)
ax.axvspan(
    xmin=fc_thresh,
    xmax=15,
    ymin=y_thresh_val,
    ymax=14,
    color="#fce4d6",
    alpha=0.5,
    zorder=0,
)

# Plot all background points in a uniform, unmanipulated grey color
ax.scatter(
    df[logfc_col],
    df["neglog10_p"],
    s=18,
    c="#7f7f7f",
    alpha=0.4,
    linewidths=0,
    label="Data Points",
    zorder=1,
)

# Threshold lines
ax.axvline(fc_thresh, color="black", linestyle="--", linewidth=1, zorder=2)
ax.axvline(-fc_thresh, color="black", linestyle="--", linewidth=1, zorder=2)
ax.axhline(y_thresh_val, color="black", linestyle="--", linewidth=1, zorder=2)

# Plot and label highlights
for _, r in highlight.iterrows():
    label_text = (
        r["gene_id_str"]
        if r["gene_str"] == "-" or r["gene_str"] == "nan"
        else r["gene_str"]
    )

    log_fc_val = r[logfc_col]

    # ALWAYS Positive Fold Change Calculation (Magnitude of difference)
    abs_fc = 2 ** abs(log_fc_val)
    fold_str = f"(Fold Change: {abs_fc:.1f}x)"

    # Font color locked to Dark Blue per instructions
    val_color = "#003399"

    # Encircle the data point cleanly (transparent middle)
    ax.scatter(
        r[logfc_col],
        r["neglog10_p"],
        s=120,
        facecolors="none",
        edgecolors="black",
        linewidths=1.5,
        zorder=5,
    )

    # Place the Gene name in black text
    t1 = ax.text(
        r[logfc_col] + 0.06,
        r["neglog10_p"] + 0.04,
        label_text + " ",
        fontsize=10,
        weight="bold",
        color="black",
        zorder=6,
    )

    # Render layout to calculate text bounding boxes dynamically
    plt.draw()

    # Place the "Fold Change: X.Xx" string in dark blue immediately following the name
    ax.annotate(
        fold_str,
        xycoords=t1,
        xy=(1.0, 0.0),
        textcoords="offset points",
        xytext=(0, 0),
        fontsize=10,
        weight="bold",
        color=val_color,
        zorder=6,
    )

# Formatting
ax.set_xlabel("log2 fold change (Differentiated / Undifferentiated)")
ax.set_ylabel("-log10(q value)")
ax.set_title("Volcano plot: SH-SY5Y Differentiated vs Undifferentiated")

# Force graph frame limits
ax.set_xlim(-15, 15)
ax.set_ylim(-0.5, 14)

# Dynamic text overlays explaining the colored background zones to the viewer
ax.text(
    -8,
    13.2,
    "Higher in Differentiated\n(Upregulated here)",
    color="#385723",
    weight="bold",
    fontsize=11,
    ha="center",
)
ax.text(
    8,
    13.2,
    "Higher in Undifferentiated\n(Upregulated here)",
    color="#c65911",
    weight="bold",
    fontsize=11,
    ha="center",
)

plt.tight_layout()
plt.savefig("volcano_segmented_intuitive_Mito.png", dpi=300, bbox_inches="tight")
#plt.savefig("PV_BA_volcano_diff_vs_undiff.svg", dpi=300, bbox_inches="tight")
plt.show()
