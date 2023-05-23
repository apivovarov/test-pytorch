# %%
import torch
import torchvision
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2
from PIL import Image
torch.__version__

# %%
weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
m = maskrcnn_resnet50_fpn_v2(weights=weights)
m = m.eval()

# %%
image = Image.open("zidane.jpg")
preprocess = weights.transforms()
data = preprocess(image)
data = data.unsqueeze(0)
print(data.shape)

# %%
res = m(data)
print(res)

# %%
import time
import numpy as np
from tqdm import tqdm

import torch._dynamo
torch._dynamo.reset()

#mode = None # 58, 55
#mode = "default" # 56, 55
#mode = "max-autotune" # 57, 55
mode = "reduce-overhead" # 58, 55
print("Compiling...")
m_opt = torch.compile(m, mode=mode)
res = m_opt(data)
print("Compilation Done")

for model, desc in [(m, "maskrcnn"), (m_opt, "maskrcnn_inductor_reduce-overhead")]:
    N = 3; i = 0
    while i < N:
        res = model(data)
        i += 1

    N = 10; i = 0
    TT = []
    while i < N:
        t0 = time.time()
        res = model(data)
        dur = (time.time() - t0) * 1000
        TT.append(dur)
        i += 1

    print(f"{desc},{np.mean(TT):.3f},{np.percentile(TT, 50):.3f}")
    with open("results.csv","a") as f:
        print(f"{desc},{np.mean(TT):.3f},{np.percentile(TT, 50):.3f}", file=f)
