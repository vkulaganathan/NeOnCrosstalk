import os
import pandas as pd

# 1. Load the metadata using tab separator
# If this gives an error, change sep='\t' to sep=','
try:
    df = pd.read_csv('gdc_samplesheet.tsv', sep='\t')
except Exception:
    df = pd.read_csv('gdc_samplesheet.tsv')

# 2. Path to your files
path = 'raw_counts'

print(f"Checking for files in {path}...")

# Counter for success
count = 0

for index, row in df.iterrows():
    # Use 'File Name' from your sheet to find the file
    old_filename = row['File Name']
    # Use 'Sample ID' to create the new name (prevents overwriting duplicates)
    new_filename = f"{row['Sample ID']}.tsv"
    
    old_path = os.path.join(path, old_filename)
    new_path = os.path.join(path, new_filename)
    
    # Check if the file exists
    if os.path.exists(old_path):
        os.rename(old_path, new_path)
        print(f"Renamed: {old_filename} ---> {new_filename}")
        count += 1
    else:
        # If the filename in the sheet is slightly different, try to find it
        # by matching just the start of the string
        file_id_prefix = old_filename.split('.')[0]
        found_backup = False
        for f in os.listdir(path):
            if f.startswith(file_id_prefix):
                os.rename(os.path.join(path, f), new_path)
                print(f"Renamed (partial match): {f} ---> {new_filename}")
                count += 1
                found_backup = True
                break
        
        if not found_backup:
            print(f"NOT FOUND: {old_filename}")

print(f"\nFinished! Successfully renamed {count} files.")
