from fastcospy import FastCOSPYEmulator

def test_model_initialization():
    model = FastCOSPYEmulator()
    assert model is not None