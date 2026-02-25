from fastcospy.utils import get_cache_dir, clear_cache
from pathlib import Path

def test_get_cache_dir():
    cache_dir = get_cache_dir()
    assert cache_dir.exists()
    assert cache_dir.is_dir()

def test_clear_cache_removes_dir():
    cache_dir = get_cache_dir()
    dummy_file = cache_dir / "dummy.pt"
    dummy_file.write_text("123")
    assert dummy_file.exists()

    clear_cache()
    assert not cache_dir.exists()

    clear_cache()
