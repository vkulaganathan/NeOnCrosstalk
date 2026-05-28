import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the dataset
file_name = 'NBL_GBM_TCGA_DataAnalysisMatrix.tsv'


df = pd.read_csv(file_name, sep='\t')

# 2. Preparation for plotting
# Avoid log(0) by adding a tiny epsilon if p-value is 0
df['-log10P'] = -np.log10(df['NBL_vs_GBM_Pvalue'] + 1e-100)

# Genes to label
genes_to_label = ["STAT1", "STAT3", "STAT4", "STAT2", "STAT5B", "STAT5A", "STAT6", "MYC", "MYB", "RAF1", "LZTR1", "CUL3", "OLIG1", "OLIG2", "SOX2", "SOX8", "POU3F3", "POU3F2", "POU3F4"]


# 3. Initialize the plot
plt.figure(figsize=(10, 8))

# Scatter plot for all genes (Background in light grey)
plt.scatter(df['NBL_vs_GBM_Log2FC'], df['-log10P'], 
            color='lightgrey', alpha=0.4, s=20, label='Other Genes')

# 4. Filter and Color the Target Genes
labeled_df = df[df['Gene Symbol'].isin(genes_to_label)].copy()

if not labeled_df.empty:
    # Upregulated (Log2FC > 0) -> Red
    up = labeled_df[labeled_df['NBL_vs_GBM_Log2FC'] > 0]
    plt.scatter(up['NBL_vs_GBM_Log2FC'], up['-log10P'], 
                color='red', s=100, edgecolors='black', label='Target: Upregulated (NBL high)')
    
    # Downregulated (Log2FC < 0) -> Green
    down = labeled_df[labeled_df['NBL_vs_GBM_Log2FC'] < 0]
    plt.scatter(down['NBL_vs_GBM_Log2FC'], down['-log10P'], 
                color='green', s=100, edgecolors='black', label='Target: Downregulated (GBM high)')

    # Add text labels for all target genes
    for i, row in labeled_df.iterrows():
        plt.annotate(row['Gene Symbol'], 
                     (row['NBL_vs_GBM_Log2FC'], row['-log10P']), 
                     textcoords="offset points", xytext=(5, 5), 
                     ha='left', fontsize=10, fontweight='bold')
else:
    print("Note: Target genes not found in this specific subset of data.")

# 5. Add Threshold Lines
# Vertical lines for 2-fold change (Log2FC = 1 and -1)
plt.axvline(x=1, color='black', linestyle='--', linewidth=1.2, label='2-fold threshold')
plt.axvline(x=-1, color='black', linestyle='--', linewidth=1.2)

# Horizontal line for significance (P=0.05)
plt.axhline(y=-np.log10(0.05), color='blue', linestyle=':', linewidth=1.2, label='P = 0.05')

# 6. Formatting and Labels
plt.xlabel('Log2 Fold Change (NBL vs GBM)', fontsize=12)
plt.ylabel('-Log10 P-value', fontsize=12)
plt.title('Volcano Plot: NBL vs GBM with Mito Gene Labeling', fontsize=14)
plt.legend(loc='upper right', frameon=True)
plt.grid(alpha=0.2)

# 7. Save and display
plt.tight_layout()
plt.savefig('NBL_vs_GBM_VolcanoPlot_Colored_MAPK.png', dpi=300)
plt.show()
