# %%
import torch
from transformers import BeitImageProcessor, BeitForImageClassification
from PIL import Image
import requests
from test_model import TestModel

class ImageClassifierWrapper(torch.nn.Module):
    def __init__(self, m) -> None:
        super().__init__()
        self.m = m
    def forward(self, pixel_values):
        res = self.m(pixel_values)
        return res.logits

# %%
class TestBeit(TestModel):
    def __init__(self) -> None:
        super().__init__()
        self.m_name = "beit-base-patch16-224"
        url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
        image = Image.open(requests.get(url, stream=True).raw)
        processor = BeitImageProcessor.from_pretrained('microsoft/beit-base-patch16-224')
        self.model = BeitForImageClassification.from_pretrained('microsoft/beit-base-patch16-224')
        inputs = processor(images=image, return_tensors="pt")
        self.data = inputs["pixel_values"]

        self.N_warmup = 10
        self.N_test = 50
        res = self.model(self.data)
        print("Optimizing for inference")
        self.traced_model = torch.jit.trace(ImageClassifierWrapper(self.model), self.data)
        print("Optimizing for inference done")

    @torch.inference_mode()
    def call_baseline_model(self):
        return self.traced_model(self.data)

# %%
test = TestBeit()
test.run_test()
