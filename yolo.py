# %%
import torch

# Model
m_name = 'yolov5s'
m = torch.hub.load('ultralytics/yolov5', m_name, pretrained=True)
m.eval()
# Images
# https://ultralytics.com/images/zidane.jpg
data = ['zidane.jpg']  # batch of images

# Inference
results = m(data)

print(results.pred)
# Results
#results.print()
#results.show()

#results.xyxy[0]  # img1 predictions (tensor)
#results.pandas().xyxy[0]

# %%
import time
import numpy as np
from tqdm import tqdm

import torch._dynamo
torch._dynamo.reset()
torch._dynamo.config.verbose=True

@torch.inference_mode()
def evaluate(mod, inp):
    return mod(inp).pred

print("Compiling...")
#mode = None # 58, 55
#mode = "default" # 56, 55
#mode = "max-autotune" # 57, 55
mode = "reduce-overhead" # 58, 55
evaluate_opt = torch.compile(evaluate, mode=mode)
res = evaluate_opt(m, data)
print("Compilation Done")

for fun, desc in [(evaluate, m_name), (evaluate_opt, f"{m_name}_inductor_reduce-overhead")]:
    N = 10; i = 0
    while i < N:
        res = fun(m, data)
        i += 1

    N = 50; i = 0
    TT = []
    while i < N:
        t0 = time.time()
        res = fun(m, data)
        dur = (time.time() - t0) * 1000
        TT.append(dur)
        i += 1

    print(f"{desc},{np.mean(TT):.3f},{np.percentile(TT, 50):.3f}")
    with open("results.csv", "a") as f:
        print(f"{desc},{np.mean(TT):.3f},{np.percentile(TT, 50):.3f}", file=f)