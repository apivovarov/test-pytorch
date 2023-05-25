import torch
import torchvision
import time
from typing import Dict
import numpy as np
from tqdm import tqdm

class TestModel():
    def __init__(self) -> None:
        self.model = None
        self.m_name = None
        self.data = None
        self.N_warmup = 10
        self.N_test = 50

    def compile_model(self):
        print("Compiling...")
        self.compiled_model = torch.compile(self.model)
        res = self.compiled_model(self.data)
        print("Compilation Done")

    @torch.inference_mode()
    def call_baseline_model(self):
        self.model(self.data)

    def call_compiled_model(self):
        self.compiled_model(self.data)

    def _run_model(self, fun, desc):
        print(f"Running {desc} test...")
        i = 0
        while i < self.N_warmup:
            res = fun()
            i += 1

        TT = []
        i = 0
        while i < self.N_test:
            t0 = time.time()
            res = fun()
            dur = (time.time() - t0) * 1000
            TT.append(dur)
            i += 1

        avg_time = np.mean(TT)
        p50_time = np.percentile(TT, 50)
        print(f"{desc},{avg_time:.2f},{p50_time:.2f}")
        return avg_time

    def run_test(self):
        AVG=["F","F"]
        AVG[0] = self._run_model(self.call_baseline_model, self.m_name)
        try:
            self.compile_model()
            AVG[1] = self._run_model(self.call_compiled_model, f"{self.m_name}_compiled")
            print(f"{self.m_name},{AVG[1]:.2f},{AVG[0]/AVG[1]:.2f}")
            with open("results.csv","a") as f:
                print(f"{self.m_name},{AVG[1]:.2f},{AVG[0]/AVG[1]:.2f}", file=f)
        except:
            print(f"{self.m_name},F,F")
            with open("results.csv","a") as f:
                print(f"{self.m_name},F,F", file=f)


class DictToListModel(torch.nn.Module):
    def __init__(self, m: torch.nn.Module) -> None:
        super().__init__()
        self.m = m
    def forward(self, x: torch.Tensor):
        res: Dict = self.m.forward(x)
        return list(res.values())


class ListDictToListModel(torch.nn.Module):
    def __init__(self, m: torch.nn.Module) -> None:
        super().__init__()
        self.m = m
    def forward(self, x: torch.Tensor):
        res: Dict = self.m.forward(x)
        return list(res[0].values())
