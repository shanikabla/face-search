import io

import pytest
from PIL import Image

from face_search.images import ImageRejected, normalize


def encode(img: Image.Image, fmt: str, **save_kwargs) -> bytes:
    buf = io.BytesIO()
    img.save(buf, fmt, **save_kwargs)
    return buf.getvalue()


def test_converts_to_jpeg_and_bounds_size(tmp_path):
    data = encode(Image.new("RGB", (2000, 1000), "red"), "PNG")
    result = normalize(data, tmp_path, max_size=1024)
    saved = Image.open(result.file_path)
    assert saved.format == "JPEG"
    assert (result.width, result.height) == saved.size == (1024, 512)  # aspect ratio kept


def test_never_upscales(tmp_path):
    result = normalize(encode(Image.new("RGB", (300, 200)), "PNG"), tmp_path, max_size=1024)
    assert (result.width, result.height) == (300, 200)


def test_transparency_becomes_white_not_black(tmp_path):
    data = encode(Image.new("RGBA", (100, 100), (0, 0, 0, 0)), "PNG")
    pixel = Image.open(normalize(data, tmp_path, 1024).file_path).getpixel((50, 50))
    assert all(channel > 245 for channel in pixel)


def test_applies_exif_rotation(tmp_path):
    img = Image.new("RGB", (400, 200))
    exif = img.getexif()
    exif[0x0112] = 6  # Orientation: "rotate 90° clockwise to display"
    result = normalize(encode(img, "JPEG", exif=exif), tmp_path, 1024)
    assert (result.width, result.height) == (200, 400)


def test_same_bytes_give_same_file(tmp_path):
    data = encode(Image.new("RGB", (100, 100), "blue"), "PNG")
    assert normalize(data, tmp_path, 1024).file_path == normalize(data, tmp_path, 1024).file_path


def test_rejects_icons(tmp_path):
    with pytest.raises(ImageRejected, match="too small"):
        normalize(encode(Image.new("RGB", (50, 50)), "PNG"), tmp_path, 1024)


def test_rejects_non_images(tmp_path):
    with pytest.raises(ImageRejected, match="unreadable"):
        normalize(b"<svg xmlns='http://www.w3.org/2000/svg'></svg>", tmp_path, 1024)
