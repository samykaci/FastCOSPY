import torch
from fastcospy import FastCOSPYEmulator

def test_forward_pass_shape():
    model = FastCOSPYEmulator(device="cpu")

    x = torch.zeros((1, 14, 21, 41))
    y = model.predict(x)

    assert isinstance(y, torch.Tensor)
    assert y.shape[0] == x.shape[0]
