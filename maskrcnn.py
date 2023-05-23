# %%
import torch
import torchvision
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2
from PIL import Image
torch.__version__

# %%
m_name = "MaskRCNN"
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

try:
    #mode = None # 58, 55
    #mode = "default" # 56, 55
    #mode = "max-autotune" # 57, 55
    mode = "reduce-overhead" # 58, 55
    print("Compiling...")
    m_opt = torch.compile(m, mode=mode)
    res = m_opt(data)
    print("Compilation Done")

    AVG=[]

    for model, desc in [(m, m_name), (m_opt, f"{m_name}_compiled")]:
        N = 5; i = 0
        while i < N:
            res = model(data)
            i += 1

        N = 20; i = 0
        TT = []
        while i < N:
            t0 = time.time()
            res = model(data)
            dur = (time.time() - t0) * 1000
            TT.append(dur)
            i += 1

        AVG.append(np.mean(TT))
        print(f"{desc},{np.mean(TT):.2f},{np.percentile(TT, 50):.2f}")

    print(f"{m_name},{AVG[1]:.2f},{AVG[0]/AVG[1]:.2f}")
    with open("results.csv","a") as f:
        print(f"{m_name},{AVG[1]:.2f},{AVG[0]/AVG[1]:.2f}", file=f)
except:
    print(f"{m_name},F,F")
    with open("results.csv","a") as f:
        print(f"{m_name},F,F", file=f)
