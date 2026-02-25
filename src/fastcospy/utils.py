from pathlib import Path
from typing import Optional

import requests
import torch

import shutil


def get_cache_dir() -> Path:
    """
    Return the directory used to cache pretrained weights.

    The directory is created if it does not exist.

    Returns
    -------
    pathlib.Path
        Path to the cache directory.
    """
    cache_dir = Path.home() / ".fastcospy"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def download_weights(url: str, filename: Optional[str] = None, force: bool = False) -> Path:
    """
    Download pretrained model weights if not already cached.

    Parameters
    ----------
    url : str
        Direct URL to the weights file.
    filename : str, optional
        Name of the local file. If None, it is inferred from the URL.
    force : bool, optional
        If True, re-download even if the file already exists.

    Returns
    -------
    pathlib.Path
        Path to the downloaded weights file.
    """
    cache_dir = get_cache_dir()

    if filename is None:
        filename = url.split("/")[-1]

    weights_path = cache_dir / filename

    if weights_path.exists() and not force:
        return weights_path

    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(weights_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    return weights_path


def load_weights(
    model: torch.nn.Module,
    weights_url: str,
    filename: Optional[str] = None,
    map_location: str = "cpu",
) -> torch.nn.Module:
    """
    Load pretrained weights into a model.

    Parameters
    ----------
    model : torch.nn.Module
        Model instance to load weights into.
    weights_url : str
        URL to the pretrained weights file.
    filename : str, optional
        Local filename for caching the weights.
    map_location : str, optional
        Device mapping for torch.load (default: "cpu").

    Returns
    -------
    torch.nn.Module
        Model with loaded weights.
    """
    weights_path = download_weights(weights_url, filename=filename)

    state_dict = torch.load(weights_path, map_location=map_location)
    model.load_state_dict(state_dict)

    return model


def clear_cache():
    """
    Delete the entire FastCOSPY cache directory, removing all downloaded weights.

    This is useful to free disk space or reset the cached models.
    Use with caution: all cached files will be permanently deleted.
    """
    cache_dir = get_cache_dir()

    if cache_dir.exists() and cache_dir.is_dir():
        shutil.rmtree(cache_dir)
        print(f"Cache cleared: {cache_dir}")
    else:
        print(f"No cache found at {cache_dir}")