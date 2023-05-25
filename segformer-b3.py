# %%
import torch
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image
import requests
from test_model import TestModel

class SegformerWrapper(torch.nn.Module):
    def __init__(self, m) -> None:
        super().__init__()
        self.m = m
    def forward(self, pixel_values):
        return self.m(pixel_values, return_dict=False)

# %%
class TestSegformer(TestModel):
    def __init__(self) -> None:
        super().__init__()
        self.m_name = "segformer-b3"
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b3-finetuned-ade-512-512")
        self.model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b3-finetuned-ade-512-512")

        inputs = feature_extractor(images=image, return_tensors="pt")
        self.data = inputs.pixel_values

        self.N_warmup = 5
        self.N_test = 30
        res = self.model(self.data)
        print("Optimizing for inference")
        self.traced_model = torch.jit.trace(SegformerWrapper(self.model), self.data)
        print("Optimizing for inference done")

    @torch.inference_mode()
    def call_baseline_model(self):
        return self.traced_model(self.data)

    def call_compiled_model(self):
        return self.compiled_model(self.data, return_dict=False)

# %%
test = TestSegformer()
test.run_test()
