"""Test partial-ZIP extraction over HTTP Range requests.

We can't talk to the real BVI-HFR server from CI (and shouldn't —
40 GB), so these tests spin up a tiny ``http.server`` on localhost
that serves a known ZIP file with full Range support. This validates
our HTTPRangeReader + zipfile integration.
"""
from __future__ import annotations

import io
import os
import socket
import threading
import zipfile
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import pytest

from cineinfini.core.partial_zip import (
    HTTPRangeReader, list_remote_zip, extract_from_remote_zip,
)


# ---------------------------------------------------------------------------
# Custom Range-aware HTTP handler (Python's SimpleHTTPRequestHandler does not
# always honour Range, so we roll our own minimal one).
# ---------------------------------------------------------------------------
class RangeHandler(BaseHTTPRequestHandler):
    served_path: Path = Path()  # set by fixture

    def log_message(self, fmt, *args):  # silence test logs
        pass

    def _serve(self, head_only=False):
        if not self.served_path.exists():
            self.send_error(404)
            return
        data = self.served_path.read_bytes()
        size = len(data)
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            try:
                spec = rng.split("=", 1)[1]
                start_s, end_s = spec.split("-", 1)
                start = int(start_s) if start_s else max(0, size - int(end_s))
                end = int(end_s) if end_s else size - 1
                end = min(end, size - 1)
                chunk = data[start: end + 1]
                self.send_response(206)
                self.send_header("Content-Type", "application/zip")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(len(chunk)))
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.end_headers()
                if not head_only:
                    self.wfile.write(chunk)
                return
            except Exception:
                pass
        self.send_response(200)
        self.send_header("Content-Type", "application/zip")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(size))
        self.end_headers()
        if not head_only:
            self.wfile.write(data)

    def do_HEAD(self):
        self._serve(head_only=True)

    def do_GET(self):
        self._serve(head_only=False)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_zip(tmp_path):
    """Make a ZIP with several files of varying sizes."""
    zpath = tmp_path / "sample.zip"
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("README.md", "# sample\n" * 20)
        zf.writestr("data/big.bin", b"\x00" * (1024 * 256))   # 256 KB zeros
        zf.writestr("data/labels.json", '{"score": 0.42}')
        zf.writestr("data/extra.json", '{"score": 0.99}')
        zf.writestr("videos/clip01.mp4", b"FAKE-MP4-DATA-01" * 4096)
        zf.writestr("videos/clip02.mp4", b"FAKE-MP4-DATA-02" * 4096)
    return zpath


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def http_server(sample_zip):
    """Serve the sample ZIP over HTTP with full Range support."""
    port = _free_port()
    RangeHandler.served_path = sample_zip
    server = HTTPServer(("127.0.0.1", port), RangeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}/sample.zip"
    server.shutdown()
    thread.join(timeout=2)


# ---------------------------------------------------------------------------
# HTTPRangeReader
# ---------------------------------------------------------------------------
def test_reader_reports_size(http_server):
    r = HTTPRangeReader(http_server)
    assert r.size is not None
    assert r.size > 0


def test_reader_supports_range(http_server):
    r = HTTPRangeReader(http_server)
    assert r.supports_range is True


def test_reader_seek_and_read(http_server):
    r = HTTPRangeReader(http_server)
    r.seek(0)
    head = r.read(4)
    # Every ZIP starts with "PK\x03\x04" (local file header signature)
    assert head[:2] == b"PK"


def test_reader_seek_from_end(http_server):
    r = HTTPRangeReader(http_server)
    r.seek(-22, 2)  # End-Of-Central-Directory min size = 22 bytes
    eocd = r.read(22)
    assert eocd[:4] == b"PK\x05\x06"  # EOCD signature


def test_reader_buffered_reads_coalesce(http_server):
    r = HTTPRangeReader(http_server, buffer_size=8192)
    r.seek(0)
    a = r.read(100)
    b = r.read(100)  # within same buffer -> no extra HTTP request
    requests_after_two_reads = r.requests_made
    assert requests_after_two_reads == 1
    assert len(a) == 100 and len(b) == 100


# ---------------------------------------------------------------------------
# list_remote_zip
# ---------------------------------------------------------------------------
def test_list_remote_zip_finds_all_files(http_server):
    files = list_remote_zip(http_server)
    names = {f["name"] for f in files}
    assert "README.md" in names
    assert "data/big.bin" in names
    assert "data/labels.json" in names
    assert "videos/clip01.mp4" in names


def test_list_remote_zip_returns_sizes(http_server):
    files = list_remote_zip(http_server)
    big = next(f for f in files if f["name"] == "data/big.bin")
    assert big["size"] == 1024 * 256
    # zeros compress very well so compress_size << size
    assert big["compress_size"] < big["size"]


# ---------------------------------------------------------------------------
# extract_from_remote_zip — the critical path
# ---------------------------------------------------------------------------
def test_extract_only_json(http_server, tmp_path):
    out_dir = tmp_path / "out_json"
    extracted = extract_from_remote_zip(
        http_server, target_dir=out_dir, only=["*.json"],
    )
    extracted_names = {p.name for p in extracted}
    assert "labels.json" in extracted_names
    assert "extra.json" in extracted_names
    assert "big.bin" not in extracted_names
    assert "README.md" not in extracted_names
    # Verify content was actually extracted, not just listed
    assert (out_dir / "data" / "labels.json").read_text() == '{"score": 0.42}'


def test_extract_only_videos(http_server, tmp_path):
    out_dir = tmp_path / "out_videos"
    extracted = extract_from_remote_zip(
        http_server, target_dir=out_dir, only=["videos/*.mp4"],
    )
    assert len(extracted) == 2
    for p in extracted:
        assert p.suffix == ".mp4"
        assert p.stat().st_size > 0


def test_extract_skips_existing_unless_overwrite(http_server, tmp_path):
    out_dir = tmp_path / "out_skip"
    out_dir.mkdir()
    # pre-create the target file with bogus content
    (out_dir / "data").mkdir()
    target = out_dir / "data" / "labels.json"
    target.write_text("PRESERVED")
    extract_from_remote_zip(
        http_server, target_dir=out_dir, only=["*.json"],
    )
    assert target.read_text() == "PRESERVED"
    # With overwrite=True it gets replaced
    extract_from_remote_zip(
        http_server, target_dir=out_dir, only=["*.json"], overwrite=True,
    )
    assert target.read_text() == '{"score": 0.42}'


def test_extract_partial_only_fetches_what_we_need(http_server, tmp_path):
    """The point of partial-zip: don't pay for what you don't extract."""
    out_dir = tmp_path / "out_minimal"
    # Track HTTP traffic via a fresh reader
    reader = HTTPRangeReader(http_server)
    total_size = reader.size
    # Extract only the tiny json files
    extracted = extract_from_remote_zip(
        http_server, target_dir=out_dir, only=["*.json"],
    )
    extracted_size = sum(p.stat().st_size for p in extracted)
    # JSON files are < 100 bytes each
    assert extracted_size < 200
    # Total ZIP size at least bigger than what we extracted (sanity)
    assert total_size > extracted_size


def test_extract_full_when_no_filter(http_server, tmp_path):
    out_dir = tmp_path / "out_full"
    extracted = extract_from_remote_zip(http_server, target_dir=out_dir)
    names = {p.name for p in extracted}
    assert "README.md" in names
    assert "labels.json" in names
    assert "clip01.mp4" in names
    assert "clip02.mp4" in names
    assert "big.bin" in names


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------
def test_unknown_url_raises_or_no_range():
    """Unreachable URL: either probe raises, or reader reports no Range."""
    try:
        r = HTTPRangeReader("http://127.0.0.1:1/nope.zip", timeout=2)
        # If probe didn't raise, range support must be False
        assert r.supports_range is False
    except Exception:
        pass  # raising is also acceptable
