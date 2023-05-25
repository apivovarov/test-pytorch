# %%
import torch
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights
from PIL import Image
from test_model import TestModel

# %%
class TestResnext50(TestModel):
    def __init__(self) -> None:
        super().__init__()
        self.m_name = "resnext50_32x4d"
        weights = ResNeXt50_32X4D_Weights.DEFAULT
        m = resnext50_32x4d(weights=weights)
        m = m.eval()
        self.model = m

        image = Image.open("cat.jpg")
        preprocess_fun = weights.transforms()
        data = preprocess_fun(image)
        self.data = data.unsqueeze(0)
        self.N_warmup = 10
        self.N_test = 50
        res = self.model(self.data)
        print("Optimizing for inference")
        self.frozen_model = torch.jit.optimize_for_inference(torch.jit.script(m))
        print("Optimizing for inference done")

    @torch.inference_mode()
    def call_baseline_model(self):
        return self.frozen_model(self.data)


# %%
test = TestResnext50()
test.run_test()
