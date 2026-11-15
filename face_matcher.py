"""Search the indexed sites for pages showing the same person as the face in an image.

Usage (from the repo root, once the pipeline has run):
    docker compose run --rm face-matcher queries/person.jpg
    docker compose run --rm face-matcher queries/person.jpg --max-distance 0.25 --json
"""
import argparse
import json
import os
import sys

import psycopg

# Silence TensorFlow's C++ start-up logs; must be set before TensorFlow is imported
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from face_search import db, faces, search  # noqa: E402  (after the env var on purpose)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", help="path to the query image")
    parser.add_argument(
        "--max-distance", type=float, default=search.DEFAULT_MAX_DISTANCE,
        help=f"max cosine distance for a match (default: {search.DEFAULT_MAX_DISTANCE}); "
             "lower = stricter",
    )
    parser.add_argument(
        "--all-faces", action="store_true",
        help="search every face in the image (default: only the largest one)",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not os.path.isfile(args.image):
        print(f"error: file not found: {args.image}", file=sys.stderr)
        return 2

    query_faces = faces.detect_and_embed(args.image)
    if not query_faces:
        print("error: no face found in the query image "
              f"(needs confidence >= {faces.MIN_CONFIDENCE} and >= {faces.MIN_FACE_PX}px)",
              file=sys.stderr)
        return 1
    # The query is "an image of a specific character": by default, the main (largest) face
    query_faces.sort(key=lambda f: f.w * f.h, reverse=True)
    if not args.all_faces:
        query_faces = query_faces[:1]

    results = []
    with db.connect() as conn:
        try:
            indexed = conn.execute(
                "SELECT count(*) FROM faces WHERE model = %s", (faces.MODEL_NAME,)
            ).fetchone()[0]
        except psycopg.errors.UndefinedTable:
            indexed = 0
        if not indexed:
            print("error: the index is empty, run the face_pipeline DAG first", file=sys.stderr)
            return 1
        for face in query_faces:
            matches = search.find_matching_pages(conn, face.embedding, faces.MODEL_NAME, args.max_distance)
            results.append((face, matches))

    if args.json:
        print(json.dumps([
            {
                "query_face": {"x": f.x, "y": f.y, "w": f.w, "h": f.h, "confidence": f.confidence},
                "pages": [
                    {"page_url": m.page_url, "similarity": round(m.similarity, 3),
                     "distance": round(m.best_distance, 3), "image_urls": m.image_urls}
                    for m in matches
                ],
            }
            for f, matches in results
        ], indent=2))
        return 0

    for face, matches in results:
        print(f"\nQuery face at x={face.x} y={face.y} ({face.w}x{face.h}px), "
              f"max distance {args.max_distance}")
        if not matches:
            print("  No page found with this person.")
            continue
        print(f"  Found on {len(matches)} page(s):\n")
        for m in matches:
            print(f"  similarity {m.similarity:.3f}  {m.page_url}")
            for url in m.image_urls:
                print(f"      image: {url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
