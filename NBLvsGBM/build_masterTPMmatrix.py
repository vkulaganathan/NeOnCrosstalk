import os
import pandas as pd

# Current directory where the script and files are
path = './'
output_file = 'master_tpm_matrix.tsv'

# Get all .tsv files, but EXCLUDE the output file if it already exists
files = [f for f in os.listdir(path) if f.endswith('.tsv') and f != output_file]

master_df = None

print(f"Processing {len(files)} files found in {path}...")

for f in files:
    sample_id = f.replace('.tsv', '')
    
    try:
        # Read GDC file: skip first row, use tab separator
        df = pd.read_csv(os.path.join(path, f), sep='\t', skiprows=1)
        
        # Remove metadata rows (N_unmapped, etc)
        df = df[~df['gene_id'].str.startswith('N_')]
        
        # Select target columns
        subset = df[['gene_id', 'gene_name', 'tpm_unstranded']]
        
        # Use Sample ID as the column header for the TPM values
        subset = subset.rename(columns={'tpm_unstranded': sample_id})
        
        if master_df is None:
            master_df = subset
        else:
            # Merge on gene_id and gene_name
            master_df = pd.merge(master_df, subset, on=['gene_id', 'gene_name'], how='outer')
            
        print(f"Successfully merged: {sample_id}")
        
    except Exception as e:
        print(f"Error processing {f}: {e}")

# Save as Tab-Separated Values (TSV)
if master_df is not None:
    master_df.to_csv(output_file, sep='\t', index=False)
    print(f"\nSUCCESS! Master matrix saved as: {output_file}")
    print(f"Total samples merged: {len(files)}")
else:
    print("No files were merged.")
