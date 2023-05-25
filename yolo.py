# %%
import torch
from test_model import TestModel

# %%
class TestYolov5s(TestModel):
    def __init__(self) -> None:
        super().__init__()
        self.m_name = 'yolov5s'
        m = torch.hub.load('ultralytics/yolov5', self.m_name, pretrained=True)
        self.model = m.eval()
        self.data = ['zidane.jpg']
        self.N_warmup = 10
        self.N_test = 50
        res = self.model(self.data)

# %%
test = TestYolov5s()
test.run_test()
