"""Download an image and normalize it: RGB JPEG, bounded size, correct orientation."""
import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

import requests
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024
# Below this, an image is an icon/logo: too small to hold a recognizable face
MIN_SIDE_PX = 64


class ImageRejected(Exception):
    """Permanent: the URL is not a usable image. Retrying won't help."""


@dataclass
class NormalizedImage:
    sha256: str
    file_path: str
    width: int
    height: int


def download(url: str, session: requests.Session) -> bytes:
    """Raises requests.RequestException on transient errors (network, 5xx, 429)."""
    with session.get(url, timeout=20, stream=True) as response:
        # 4xx (404 Not Found, 403 Forbidden...) won't change on retry, except 429 Too Many Requests
        if 400 <= response.status_code < 500 and response.status_code != 429:
            raise ImageRejected(f"HTTP {response.status_code}")
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")
        if not content_type.startswith("image/"):
            raise ImageRejected(f"not an image: Content-Type={content_type!r}")
        data = response.raw.read(MAX_DOWNLOAD_BYTES + 1, decode_content=True)
    if len(data) > MAX_DOWNLOAD_BYTES:
        raise ImageRejected("file too large")
    return data


def normalize(data: bytes, out_dir: Path, max_size: int) -> NormalizedImage:
    # Hash the original bytes: identical files from different URLs share one row
    sha256 = hashlib.sha256(data).hexdigest()
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ImageRejected(f"unreadable image: {exc}") from exc

    if min(img.size) < MIN_SIDE_PX:
        raise ImageRejected(f"too small: {img.size}")

    # Phone photos are often stored sideways with an EXIF "rotate me" tag
    img = ImageOps.exif_transpose(img)
    img = to_rgb(img)
    img.thumbnail((max_size, max_size))  # keeps aspect ratio, never upscales

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{sha256}.jpg"
    img.save(path, "JPEG", quality=90)
    return NormalizedImage(sha256, str(path), img.width, img.height)


def to_rgb(img: Image.Image) -> Image.Image:
    """JPEG has no transparency: paint transparent areas white instead of black."""
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.getchannel("A"))
        return background
    return img.convert("RGB")
