# %%
import torch
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2
from PIL import Image
from test_model import TestModel

# %%
class TestMaskRCNN(TestModel):
    def __init__(self) -> None:
        super().__init__()
        self.m_name = "MaskRCNN"
        weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        m = maskrcnn_resnet50_fpn_v2(weights=weights)
        self.model = m.eval()
        image = Image.open("zidane.jpg")
        preprocess_fun = weights.transforms()
        data = preprocess_fun(image)
        self.data = data.unsqueeze(0)
        self.N_warmup = 5
        self.N_test = 30

# %%
test = TestMaskRCNN()
test.run_test()
