from pathlib import Path
import torch
from .models import UNetLSTMAttention
from .utils import download_weights, get_cache_dir

class FastCOSPYEmulator:
    """
    Emulator for FastCOSPY: loads a pretrained U-Net + LSTM + Attention model
    for multi-energy predictions.

    Example usage:
        emulator = FastCOSPYEmulator(device="cuda")
        output = emulator.predict(input_tensor)
    """

    def __init__(self, n_channels: int = 14, device: str = "cpu", weights_url: str = None):
        """
        Initialize the emulator, download weights if necessary, and load the model.

        Parameters
        ----------
        device : str, optional (default="cpu")
            Device for inference ("cpu" or "cuda").
        weights_url : str, optional
            URL to download pretrained weights. If None, a default URL is used.
        """
        self.device = device

        if weights_url is None:
            weights_url = "https://github.com/samykaci/FastCOSPY/releases/download/v1.0.0-weights/fastcospy_weights_v1.0.0.pth"

        self.weights_path = download_weights(weights_url)
        self.model = UNetLSTMAttention(n_channels)
        self._load_weights()
        self.model.to(self.device)
        self.model.eval()

    def _load_weights(self):
        """Load weights from the cached file into the model."""
        state_dict = torch.load(self.weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run inference on input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, H, W), where C should match the model's input channels.

        Returns
        -------
        torch.Tensor
            Predicted tensor of shape (B, n_energies, H, W).
        """
        x = x.to(self.device)
        with torch.no_grad():
            output = self.model(x)
        return output
