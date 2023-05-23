# %%
import torch
import torchvision
from PIL import Image
import requests
import torchvision.transforms as T
torch.__version__

# %%
# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# %%
m_name = "detr_resnet50"
m = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
m.eval();

# %%
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
im = Image.open(requests.get(url, stream=True).raw)

# %%
# mean-std normalize the input image (batch-size: 1)
data = transform(im).unsqueeze(0)

# propagate through the model
outputs = m(data)

# keep only predictions with 0.7+ confidence
probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]


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
        N = 10; i = 0
        while i < N:
            res = model(data)
            i += 1

        N = 30; i = 0
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
