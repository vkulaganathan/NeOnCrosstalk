import pandas as pd

# Load the TSV file into a DataFrame
df = pd.read_csv('1_rna_cancer_sample.tsv.gz', compression='gzip', sep='\t')

# Specify the cancer types of interest
cancer_types_of_interest = ['GBM']

# Filter the DataFrame to include only the specified cancer types
filtered_df = df[df['Cancer'].isin(cancer_types_of_interest)]

# Save the filtered DataFrame to a new TSV file
filtered_df.to_csv('GBMonly_data.tsv', sep='\t', index=False, header=True)

print("'GBMonly_data.tsv'")

