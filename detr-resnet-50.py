# %%
import torch
import torchvision.transforms as T
from PIL import Image
import requests
from test_model import DictToListModel, TestModel

# %%
class TestDetrResnet50(TestModel):
    def __init__(self) -> None:
        super().__init__()

        # standard PyTorch mean-std input image normalization
        transform = T.Compose([
            T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.m_name = "detr_resnet50"
        m = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
        self.model = m.eval()

        url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
        im = Image.open(requests.get(url, stream=True).raw)
        # mean-std normalize the input image (batch-size: 1)
        self.data = transform(im).unsqueeze(0)

        self.N_warmup = 5
        self.N_test = 30
        res = self.model(self.data)
        print("Optimizing for inference")
        self.traced_model = torch.jit.trace(DictToListModel(m), self.data)
        #self.frozen_model = torch.jit.optimize_for_inference(torch.jit.script(m))
        print("Optimizing for inference done")

    @torch.inference_mode()
    def call_baseline_model(self):
        return self.traced_model(self.data)

# %%
test = TestDetrResnet50()
test.run_test()
