# %%
import csv
import pandas as pd
from tabulate import tabulate

headers = ["Model", "AVG (ms)", "Speedup"]
df = pd.read_csv('results.csv', header=None)

print(tabulate(df, headers = headers, tablefmt = 'simple'))

# %%
