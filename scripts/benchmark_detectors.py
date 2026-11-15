"""Compare DeepFace's face detectors on a random sample of the indexed images.

    docker compose run --rm --entrypoint python face-matcher scripts/benchmark_detectors.py [--n 40]

Prints faces found / speed per detector and writes data/bench/detectors.jpg: the images
where the detectors disagree most, one column per detector (counts alone hide false positives).
"""
import argparse
import logging
import os
import random
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from deepface import DeepFace  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from face_search import db  # noqa: E402

BACKENDS = ["opencv", "yunet", "mtcnn", "retinaface"]


def detect(path: str, backend: str) -> list[tuple]:
    out = DeepFace.extract_faces(path, detector_backend=backend, enforce_detection=False)
    return [(f["facial_area"]["x"], f["facial_area"]["y"], f["facial_area"]["w"],
             f["facial_area"]["h"], f["confidence"]) for f in out if f["confidence"] > 0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    logging.disable(logging.WARNING)

    with db.connect() as conn:
        paths = [r[0] for r in conn.execute("SELECT file_path FROM images ORDER BY id")]
    sample = random.Random(args.seed).sample(paths, min(args.n, len(paths)))

    results = {}
    print(f"{'detector':11s} {'faces':>5s} {'imgs w/ face':>12s} {'<40px':>5s} {'s/img':>6s}")
    for backend in BACKENDS:
        detect(sample[0], backend)  # warm-up: model loading / weights download
        start = time.time()
        results[backend] = {p: detect(p, backend) for p in sample}
        per_img = (time.time() - start) / len(sample)
        found = results[backend].values()
        print(f"{backend:11s} {sum(map(len, found)):5d} {sum(bool(f) for f in found):6d}/{len(sample):<5d} "
              f"{sum(f[2] < 40 for fs in found for f in fs):5d} {per_img:6.2f}")

    def disagreement(p):
        counts = [len(results[b][p]) for b in BACKENDS]
        return max(counts) - min(counts)

    picks = sorted(sample, key=disagreement, reverse=True)[:6]
    W, H = 300, 220
    sheet = Image.new("RGB", (W * len(BACKENDS), 24 + len(picks) * (H + 4)), "white")
    draw = ImageDraw.Draw(sheet)
    for j, backend in enumerate(BACKENDS):
        draw.text((j * W + 8, 6), backend, fill="black")
        for i, p in enumerate(picks):
            img = Image.open(p).convert("RGB")
            scale = min(W / img.width, H / img.height)
            img = img.resize((int(img.width * scale), int(img.height * scale)))
            d = ImageDraw.Draw(img)
            for x, y, w, h, conf in results[backend][p]:
                d.rectangle([x * scale, y * scale, (x + w) * scale, (y + h) * scale], outline="red", width=3)
                d.text((x * scale + 2, y * scale + 2), f"{conf:.2f}", fill="yellow")
            sheet.paste(img, (j * W, 24 + i * (H + 4)))
    os.makedirs("data/bench", exist_ok=True)
    sheet.save("data/bench/detectors.jpg", quality=85)
    print("Visual comparison: data/bench/detectors.jpg")


if __name__ == "__main__":
    main()
