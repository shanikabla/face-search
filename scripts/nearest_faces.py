"""Show the K nearest indexed faces for each query image, whatever their distance.

    docker compose run --rm --entrypoint python face-matcher scripts/nearest_faces.py queries/*.jpg

Used to calibrate search.DEFAULT_MAX_DISTANCE: prints the distances and writes
data/bench/nearest.jpg (one row per query: the query face, then its nearest faces)
so you can see where true matches stop and other people start.
"""
import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from PIL import Image, ImageDraw  # noqa: E402

from face_search import db, faces  # noqa: E402

NEAREST_SQL = """
SELECT i.file_path, f.x, f.y, f.w, f.h, f.embedding <=> %s AS distance
FROM faces f JOIN images i ON i.id = f.image_id
WHERE f.model = %s
ORDER BY distance
LIMIT %s
"""


def crop(path: str, x: int, y: int, w: int, h: int, size: int) -> Image.Image:
    return Image.open(path).convert("RGB").crop((x, y, x + w, y + h)).resize((size, size))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+")
    parser.add_argument("-k", type=int, default=12)
    args = parser.parse_args()

    T = 120
    sheet = Image.new("RGB", ((args.k + 1) * T, len(args.images) * (T + 16)), "white")
    draw = ImageDraw.Draw(sheet)
    with db.connect() as conn:
        for i, query in enumerate(args.images):
            y0 = i * (T + 16)
            found = faces.detect_and_embed(query)
            if not found:
                print(f"{query}: no face")
                continue
            face = max(found, key=lambda f: f.w * f.h)
            rows = conn.execute(NEAREST_SQL, (face.embedding, faces.MODEL_NAME, args.k)).fetchall()
            print(f"{os.path.basename(query):24s}", " ".join(f"{r[5]:.3f}" for r in rows))
            sheet.paste(crop(query, face.x, face.y, face.w, face.h, T - 4), (0, y0))
            draw.text((2, y0 + T - 4), "QUERY " + os.path.basename(query)[:14], fill="blue")
            for k, (path, x, y, w, h, dist) in enumerate(rows, start=1):
                sheet.paste(crop(path, x, y, w, h, T - 4), (k * T, y0))
                draw.text((k * T + 2, y0 + T - 4), f"d={dist:.3f}", fill="black")
    os.makedirs("data/bench", exist_ok=True)
    sheet.save("data/bench/nearest.jpg", quality=85)
    print("Contact sheet: data/bench/nearest.jpg")


if __name__ == "__main__":
    main()
