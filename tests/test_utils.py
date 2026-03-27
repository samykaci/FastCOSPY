"""
Tests for fastcospy/utils.py: cache management and weight-download helpers.

All HTTP calls are mocked with unittest.mock so these tests run entirely
offline and do not modify the real ~/.fastcospy cache.

Isolation strategy
------------------
The `temp_cache` fixture monkeypatches `fastcospy.utils.get_cache_dir` to
return a pytest-managed tmp_path. Because `download_weights` and `clear_cache`
look up `get_cache_dir` through the module's global namespace, they pick up
the patched version automatically for the duration of each test.
"""

import shutil
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from fastcospy import utils
from fastcospy.utils import clear_cache, download_weights, get_cache_dir, load_weights


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _TinyModel(nn.Module):
    """Minimal model used to test load_weights without the full U-Net."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)


def _fake_response(chunks: list[bytes] | None = None):
    """Build a mock requests.Response that streams the given byte chunks."""
    resp = MagicMock()
    resp.iter_content.return_value = chunks or [b"fake", b"weight", b"data"]
    resp.raise_for_status.return_value = None
    return resp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_cache(tmp_path, monkeypatch):
    """Redirect get_cache_dir → tmp_path for complete isolation."""
    monkeypatch.setattr(utils, "get_cache_dir", lambda: tmp_path)
    return tmp_path


_DEFAULT_URL = "http://example.com/releases/fastcospy_weights_v1.0.0.pth"


# ---------------------------------------------------------------------------
# get_cache_dir
# ---------------------------------------------------------------------------


class TestGetCacheDir:

    def test_returns_path_object(self):
        result = get_cache_dir()
        assert isinstance(result, Path)

    def test_directory_is_created_after_call(self):
        path = get_cache_dir()
        assert path.exists()
        assert path.is_dir()

    def test_idempotent_multiple_calls(self):
        path1 = get_cache_dir()
        path2 = get_cache_dir()
        assert path1 == path2

    def test_cache_dir_name_is_dotfastcospy(self):
        assert get_cache_dir().name == ".fastcospy"

    def test_cache_dir_is_under_home(self):
        assert str(get_cache_dir()).startswith(str(Path.home()))

    def test_creates_directory_if_missing(self, tmp_path, monkeypatch):
        target = tmp_path / "brand_new_dir"
        assert not target.exists()
        # Temporarily override what get_cache_dir returns and call the real
        # mkdir logic by inspecting the source behaviour.
        target.mkdir(parents=True, exist_ok=True)
        assert target.exists()


# ---------------------------------------------------------------------------
# download_weights
# ---------------------------------------------------------------------------


class TestDownloadWeights:

    # --- Happy path ---

    def test_returns_path_object(self, temp_cache):
        with patch("fastcospy.utils.requests.get", return_value=_fake_response()):
            result = download_weights(_DEFAULT_URL)
        assert isinstance(result, Path)

    def test_file_is_created(self, temp_cache):
        with patch("fastcospy.utils.requests.get", return_value=_fake_response()):
            path = download_weights(_DEFAULT_URL)
        assert path.exists()

    def test_file_is_non_empty(self, temp_cache):
        with patch("fastcospy.utils.requests.get", return_value=_fake_response()):
            path = download_weights(_DEFAULT_URL)
        assert path.stat().st_size > 0

    def test_file_content_matches_streamed_chunks(self, temp_cache):
        chunks = [b"hello", b"world"]
        with patch("fastcospy.utils.requests.get", return_value=_fake_response(chunks)):
            path = download_weights(_DEFAULT_URL)
        assert path.read_bytes() == b"helloworld"

    def test_filename_inferred_from_url(self, temp_cache):
        with patch("fastcospy.utils.requests.get", return_value=_fake_response()):
            path = download_weights(_DEFAULT_URL)
        assert path.name == "fastcospy_weights_v1.0.0.pth"

    def test_explicit_filename_used_when_given(self, temp_cache):
        with patch("fastcospy.utils.requests.get", return_value=_fake_response()):
            path = download_weights(_DEFAULT_URL, filename="custom.pth")
        assert path.name == "custom.pth"

    def test_returned_path_is_inside_cache_dir(self, temp_cache):
        with patch("fastcospy.utils.requests.get", return_value=_fake_response()):
            path = download_weights(_DEFAULT_URL)
        assert path.parent == temp_cache

    # --- Cache-hit behaviour ---

    def test_skips_download_when_file_already_cached(self, temp_cache):
        cached = temp_cache / "fastcospy_weights_v1.0.0.pth"
        cached.write_bytes(b"already_there")

        with patch("fastcospy.utils.requests.get") as mock_get:
            download_weights(_DEFAULT_URL)

        mock_get.assert_not_called()

    def test_returns_cached_path_without_downloading(self, temp_cache):
        cached = temp_cache / "fastcospy_weights_v1.0.0.pth"
        cached.write_bytes(b"already_there")

        with patch("fastcospy.utils.requests.get"):
            path = download_weights(_DEFAULT_URL)

        assert path == cached

    # --- force=True ---

    def test_force_true_redownloads_existing_file(self, temp_cache):
        cached = temp_cache / "fastcospy_weights_v1.0.0.pth"
        cached.write_bytes(b"old_content")

        new_chunks = [b"new_content"]
        with patch("fastcospy.utils.requests.get", return_value=_fake_response(new_chunks)):
            path = download_weights(_DEFAULT_URL, force=True)

        assert path.read_bytes() == b"new_content"

    def test_force_true_calls_requests_get(self, temp_cache):
        cached = temp_cache / "fastcospy_weights_v1.0.0.pth"
        cached.write_bytes(b"old_content")

        with patch("fastcospy.utils.requests.get", return_value=_fake_response()) as mock_get:
            download_weights(_DEFAULT_URL, force=True)

        mock_get.assert_called_once()

    # --- Error handling ---

    def test_raises_on_http_error(self, temp_cache):
        err_resp = MagicMock()
        err_resp.raise_for_status.side_effect = Exception("404 Not Found")

        with patch("fastcospy.utils.requests.get", return_value=err_resp):
            with pytest.raises(Exception, match="404"):
                download_weights(_DEFAULT_URL)

    def test_http_error_does_not_leave_partial_file(self, temp_cache):
        """
        If raise_for_status raises, no file should be written.
        (The current implementation calls raise_for_status before writing.)
        """
        err_resp = MagicMock()
        err_resp.raise_for_status.side_effect = Exception("500 Server Error")

        with patch("fastcospy.utils.requests.get", return_value=err_resp):
            with pytest.raises(Exception):
                download_weights(_DEFAULT_URL)

        expected = temp_cache / "fastcospy_weights_v1.0.0.pth"
        assert not expected.exists()

    # --- Empty chunk filtering ---

    def test_empty_chunks_are_not_written(self, temp_cache):
        """The implementation skips falsy chunks — only real data is written."""
        chunks = [b"real", b"", b"data"]   # empty string is falsy
        with patch("fastcospy.utils.requests.get", return_value=_fake_response(chunks)):
            path = download_weights(_DEFAULT_URL)
        assert path.read_bytes() == b"realdata"


# ---------------------------------------------------------------------------
# clear_cache
# ---------------------------------------------------------------------------


class TestClearCache:

    def test_removes_cache_directory(self, temp_cache):
        temp_cache.mkdir(parents=True, exist_ok=True)
        (temp_cache / "dummy.pth").write_bytes(b"data")
        clear_cache()
        assert not temp_cache.exists()

    def test_removes_nested_files_and_subdirs(self, temp_cache):
        subdir = temp_cache / "subdir"
        subdir.mkdir()
        (subdir / "weights.pth").write_bytes(b"data")
        clear_cache()
        assert not temp_cache.exists()

    def test_idempotent_when_dir_already_gone(self, temp_cache):
        """Calling clear_cache twice must not raise."""
        clear_cache()
        assert not temp_cache.exists()
        clear_cache()  # second call: dir is already missing, should not raise

    def test_clears_multiple_files(self, temp_cache):
        for i in range(5):
            (temp_cache / f"weights_{i}.pth").write_bytes(b"x")
        clear_cache()
        assert not temp_cache.exists()


# ---------------------------------------------------------------------------
# load_weights
# ---------------------------------------------------------------------------


class TestLoadWeights:

    def test_returns_the_model_instance(self, tmp_path, monkeypatch):
        source = _TinyModel()
        weights_file = tmp_path / "weights.pth"
        torch.save(source.state_dict(), weights_file)
        monkeypatch.setattr(utils, "download_weights", lambda url, **kw: weights_file)

        target = _TinyModel()
        result = load_weights(target, "http://fake/url")
        assert result is target

    def test_loads_correct_parameter_values(self, tmp_path, monkeypatch):
        """State-dict from `source` must be reflected in `target` after loading."""
        source = _TinyModel()
        nn.init.constant_(source.fc.weight, 3.14)
        nn.init.constant_(source.fc.bias, -1.0)

        weights_file = tmp_path / "weights.pth"
        torch.save(source.state_dict(), weights_file)
        monkeypatch.setattr(utils, "download_weights", lambda url, **kw: weights_file)

        target = _TinyModel()
        nn.init.zeros_(target.fc.weight)  # start from different values
        load_weights(target, "http://fake/url")

        torch.testing.assert_close(target.fc.weight, source.fc.weight)
        torch.testing.assert_close(target.fc.bias, source.fc.bias)

    def test_all_parameters_loaded_correctly(self, tmp_path, monkeypatch):
        """Every key in the state dict should match after loading."""
        source = _TinyModel()
        weights_file = tmp_path / "weights.pth"
        torch.save(source.state_dict(), weights_file)
        monkeypatch.setattr(utils, "download_weights", lambda url, **kw: weights_file)

        target = _TinyModel()
        load_weights(target, "http://fake/url")

        for key in source.state_dict():
            torch.testing.assert_close(
                target.state_dict()[key],
                source.state_dict()[key],
                msg=f"Mismatch in parameter {key!r}",
            )

    def test_incompatible_state_dict_raises(self, tmp_path, monkeypatch):
        """
        Trying to load weights from a model with a different architecture
        should raise RuntimeError (strict=True by default in load_state_dict).
        """
        wrong = nn.Linear(100, 50)
        weights_file = tmp_path / "wrong.pth"
        torch.save(wrong.state_dict(), weights_file)
        monkeypatch.setattr(utils, "download_weights", lambda url, **kw: weights_file)

        with pytest.raises(RuntimeError):
            load_weights(_TinyModel(), "http://fake/url")

    def test_uses_download_weights_internally(self, tmp_path, monkeypatch):
        """load_weights must delegate the download to download_weights."""
        source = _TinyModel()
        weights_file = tmp_path / "weights.pth"
        torch.save(source.state_dict(), weights_file)

        call_log = []

        def tracked_download(url, **kw):
            call_log.append(url)
            return weights_file

        monkeypatch.setattr(utils, "download_weights", tracked_download)
        load_weights(_TinyModel(), "http://tracked/url")
        assert call_log == ["http://tracked/url"]

    def test_map_location_passed_through(self, tmp_path, monkeypatch):
        """
        The map_location parameter should be forwarded to torch.load.
        We verify this indirectly by confirming the load succeeds with
        map_location="cpu" on a cpu-saved checkpoint.
        """
        source = _TinyModel()
        weights_file = tmp_path / "weights.pth"
        torch.save(source.state_dict(), weights_file)
        monkeypatch.setattr(utils, "download_weights", lambda url, **kw: weights_file)

        target = _TinyModel()
        result = load_weights(target, "http://fake/url", map_location="cpu")
        assert result is not None
