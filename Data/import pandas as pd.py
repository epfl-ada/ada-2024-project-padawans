import pandas as pd

# Load the .txt file as a DataFrame
df = pd.read_csv('plot_summaries.txt', delimiter='\t')  # Use '\t' for tab-separated, ',' for comma-separated
print(df.head())
