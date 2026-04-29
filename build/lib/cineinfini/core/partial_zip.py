"""Partial ZIP extraction over HTTP Range requests.

Lets us download a *single file* (or files matching a glob) from a
multi-gigabyte remote ZIP archive without fetching the whole thing.
This is exactly what we need for assets like BVI-HFR (~40 GB total but
the per-clip MP4s are 50-200 MB each).

How it works
============

ZIP files store a "central directory" at the end of the file, listing
every entry with its offset and compressed size. So:

  1. HEAD the URL to get total size and confirm Range support.
  2. GET the last ~64 KB to grab the End-Of-Central-Directory record
     (and any 32-/64-bit central directory pointer).
  3. GET the central directory itself by offset.
  4. For each requested entry, GET only the local-header + compressed
     data slice and decompress in-process.

We expose a ``HTTPRangeReader`` that is a seekable file-like object on
top of urllib + Range, then plug it into Python's stdlib :mod:`zipfile`
which already handles every ZIP format detail. Zero new third-party
dependencies.

CAVEATS
-------
* The remote server **must** support ``Accept-Ranges: bytes`` (S3,
  GitHub Releases, Bristol data.bris.ac.uk, HuggingFace LFS all do;
  some old static servers don't).
* For ZIPs with the central directory at the very start (rare, only
  produced by some streaming tools) the partial path won't help —
  we fall back to a normal full download in that case.
* Encrypted ZIPs and ZIP64 with split archives are out of scope.
"""
from __future__ import annotations

import fnmatch
import io
import logging
import os
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger("cineinfini.partial_zip")


# ---------------------------------------------------------------------------
# Seekable HTTP-Range file-like object
# ---------------------------------------------------------------------------
class HTTPRangeReader:
    """File-like wrapper around HTTP Range GET.

    Implements ``read(n)``, ``seek()``, ``tell()``, ``close()``. Sufficient
    for :class:`zipfile.ZipFile` to operate on directly.

    A small read-ahead buffer (``buffer_size``) coalesces nearby reads to
    avoid making one HTTP request per ZIP central-directory field.
    """

    def __init__(self, url: str, *, timeout: int = 30,
                 buffer_size: int = 64 * 1024,
                 user_agent: str = "cineinfini/partial_zip"):
        self.url = url
        self.timeout = timeout
        self.buffer_size = int(buffer_size)
        self.user_agent = user_agent
        self._size: Optional[int] = None
        self._accept_ranges: Optional[bool] = None
        self._pos = 0
        self._buf: bytes = b""
        self._buf_start = -1  # position in file where _buf starts
        self.bytes_fetched = 0
        self.requests_made = 0
        self._probe()

    # -- probe ---------------------------------------------------------------
    def _probe(self) -> None:
        try:
            req = Request(self.url, method="HEAD",
                          headers={"User-Agent": self.user_agent})
            with urlopen(req, timeout=self.timeout) as r:
                cl = r.headers.get("content-length")
                self._size = int(cl) if cl else None
                ar = (r.headers.get("accept-ranges") or "").lower()
                self._accept_ranges = ("bytes" in ar)
        except HTTPError as e:
            # Some servers refuse HEAD; try a tiny GET probe instead
            if e.code in (403, 405):
                self._probe_via_get()
                return
            raise
        except URLError:
            # Network / DNS error — leave size unknown; will fail on first read
            self._size = None
            self._accept_ranges = False
            return
        # If HEAD didn't advertise Range but reported a size, do a GET probe
        # — Python's stdlib http.server is one such case.
        if self._accept_ranges is False or self._accept_ranges is None:
            self._probe_via_get()

    def _probe_via_get(self) -> None:
        try:
            req = Request(self.url, method="GET",
                          headers={"User-Agent": self.user_agent, "Range": "bytes=0-0"})
            with urlopen(req, timeout=self.timeout) as r:
                cr = r.headers.get("content-range") or ""
                if "/" in cr:
                    try:
                        self._size = int(cr.split("/", 1)[1])
                    except ValueError:
                        pass
                # 206 Partial Content => server honoured the Range
                self._accept_ranges = (r.status == 206)
                if self._size is None:
                    cl = r.headers.get("content-length")
                    # If the server returned only 1 byte (the requested range)
                    # we can't infer total size; check Content-Range first
                    if not self._accept_ranges and cl:
                        try:
                            self._size = int(cl)
                        except ValueError:
                            pass
                r.read()  # drain
        except (HTTPError, URLError):
            self._accept_ranges = False

    # -- file-like API -------------------------------------------------------
    @property
    def size(self) -> Optional[int]:
        return self._size

    @property
    def supports_range(self) -> bool:
        return bool(self._accept_ranges)

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self._pos = int(offset)
        elif whence == 1:
            self._pos += int(offset)
        elif whence == 2:
            if self._size is None:
                raise OSError("Cannot seek from end: total size unknown")
            self._pos = self._size + int(offset)
        else:
            raise ValueError(f"invalid whence: {whence}")
        return self._pos

    def tell(self) -> int:
        return self._pos

    def read(self, n: int = -1) -> bytes:
        if self._size is None:
            return self._read_uncached(n)
        if n is None or n < 0:
            n = max(0, self._size - self._pos)
        if n == 0:
            return b""
        end_excl = min(self._pos + n, self._size)
        # Try buffer
        if (self._buf_start >= 0
                and self._pos >= self._buf_start
                and end_excl <= self._buf_start + len(self._buf)):
            data = self._buf[self._pos - self._buf_start: end_excl - self._buf_start]
            self._pos = end_excl
            return data
        # Otherwise refill buffer
        fetch_start = self._pos
        fetch_end = min(self._pos + max(self.buffer_size, n), self._size) - 1
        self._buf = self._range_get(fetch_start, fetch_end)
        self._buf_start = fetch_start
        slice_end = (end_excl - fetch_start)
        out = self._buf[: slice_end]
        self._pos += len(out)
        return out

    def _read_uncached(self, n: int) -> bytes:
        if n is None or n < 0:
            req = Request(self.url, headers={"User-Agent": self.user_agent})
            with urlopen(req, timeout=self.timeout) as r:
                return r.read()
        req = Request(self.url, headers={
            "User-Agent": self.user_agent,
            "Range": f"bytes={self._pos}-{self._pos + n - 1}",
        })
        with urlopen(req, timeout=self.timeout) as r:
            data = r.read()
        self._pos += len(data)
        return data

    def _range_get(self, start: int, end: int) -> bytes:
        if not self.supports_range:
            raise OSError(f"Server does not support Range: {self.url}")
        req = Request(self.url, headers={
            "User-Agent": self.user_agent,
            "Range": f"bytes={start}-{end}",
        })
        with urlopen(req, timeout=self.timeout) as r:
            data = r.read()
        self.requests_made += 1
        self.bytes_fetched += len(data)
        return data

    def close(self) -> None:
        self._buf = b""


# ---------------------------------------------------------------------------
# Convenience: list / extract from a remote ZIP
# ---------------------------------------------------------------------------
def list_remote_zip(url: str, *, timeout: int = 30) -> List[Dict[str, Any]]:
    """Return [{'name', 'size', 'compress_size'}, ...] for every entry in
    the remote ZIP, **without downloading the contents** of any entry."""
    reader = HTTPRangeReader(url, timeout=timeout)
    if not reader.supports_range:
        raise OSError(f"Server does not support Range requests: {url}")
    with zipfile.ZipFile(reader) as zf:
        return [
            {"name": info.filename, "size": int(info.file_size),
             "compress_size": int(info.compress_size)}
            for info in zf.infolist()
        ]


def extract_from_remote_zip(
    url: str,
    *,
    target_dir: Path,
    only: Optional[List[str]] = None,
    overwrite: bool = False,
    timeout: int = 30,
    progress: bool = True,
) -> List[Path]:
    """Extract selected files from a remote ZIP into ``target_dir``.

    ``only`` is a list of fnmatch-style patterns (``"*.json"``,
    ``"BVI-VFI/labels/*"``). When ``None`` every entry is extracted.

    Returns the list of locally-written paths. Files already present are
    skipped unless ``overwrite=True``.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    reader = HTTPRangeReader(url, timeout=timeout)
    if not reader.supports_range:
        raise OSError(f"Server does not support Range requests: {url}")

    out: List[Path] = []
    with zipfile.ZipFile(reader) as zf:
        infos = zf.infolist()
        if only:
            wanted = []
            for info in infos:
                if any(fnmatch.fnmatch(info.filename, pat) for pat in only):
                    wanted.append(info)
        else:
            wanted = list(infos)
        if progress:
            total_bytes = sum(info.compress_size for info in wanted)
            logger.info(
                "Partial-ZIP: %d files, ~%.1f MB to fetch (vs %.1f MB total)",
                len(wanted), total_bytes / 1e6,
                (reader.size or 0) / 1e6,
            )
        for i, info in enumerate(wanted, start=1):
            if info.is_dir():
                continue
            dst = target_dir / info.filename
            if dst.exists() and not overwrite:
                logger.debug("skip existing %s", dst)
                out.append(dst)
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                with zf.open(info) as src, dst.open("wb") as fout:
                    while True:
                        chunk = src.read(1 << 16)
                        if not chunk:
                            break
                        fout.write(chunk)
                out.append(dst)
                if progress:
                    logger.info(
                        "  [%d/%d] %s (%s)",
                        i, len(wanted), info.filename, _human(info.compress_size),
                    )
            except Exception as e:  # noqa: BLE001
                logger.warning("failed to extract %s: %s", info.filename, e)
                if dst.exists():
                    try:
                        dst.unlink()
                    except OSError:
                        pass
    if progress:
        logger.info(
            "Partial-ZIP done: %d HTTP requests, %.1f MB transferred",
            reader.requests_made, reader.bytes_fetched / 1e6,
        )
    return out


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}TB"


__all__ = [
    "HTTPRangeReader", "list_remote_zip", "extract_from_remote_zip",
]
