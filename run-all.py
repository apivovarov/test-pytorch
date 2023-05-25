# %% Run All perf tests
import subprocess

interpr = "python3"
MODELS = [
  "detr-resnet-50.py",
  "vit_b_32.py",
  "resnet50.py",
  "resnext50.py",
  "swin_v2_b.py",
  "maskrcnn.py",
  "yolo.py",
  "beit.py",
  "segformer-b3.py"
]

for f in MODELS:
  print("Running", f)
  subprocess.run([interpr, f])

# %% Print Results in a Table
import pandas as pd
from tabulate import tabulate

headers = ["Model", "AVG (ms)", "Speedup"]
df = pd.read_csv('results.csv', header=None)

print(tabulate(df, headers = headers, tablefmt = 'simple'))
