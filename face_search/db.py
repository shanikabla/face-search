"""Database access for the application DB (pages, images, faces)."""
import os

import psycopg
from pgvector.psycopg import register_vector

# Idempotent: safe to run at the start of every pipeline run.
SCHEMA = """
CREATE TABLE IF NOT EXISTS pages (
    id          BIGSERIAL PRIMARY KEY,
    url         TEXT NOT NULL UNIQUE,
    crawled_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One row per distinct image *content*: same bytes => same row, whatever the URL.
CREATE TABLE IF NOT EXISTS images (
    id          BIGSERIAL PRIMARY KEY,
    sha256      TEXT NOT NULL UNIQUE,
    file_path   TEXT NOT NULL,
    width       INT NOT NULL,
    height      INT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One row per distinct image *URL*. Several URLs may point to the same content.
--   rejected = permanent (not an image, too small); failed = transient, retried next run
CREATE TABLE IF NOT EXISTS image_urls (
    id          BIGSERIAL PRIMARY KEY,
    url         TEXT NOT NULL UNIQUE,
    image_id    BIGINT REFERENCES images(id),
    status      TEXT NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending', 'downloaded', 'rejected', 'failed')),
    error       TEXT
);

-- Which page shows which image URL (many-to-many).
CREATE TABLE IF NOT EXISTS page_image_urls (
    page_id       BIGINT NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
    image_url_id  BIGINT NOT NULL REFERENCES image_urls(id) ON DELETE CASCADE,
    PRIMARY KEY (page_id, image_url_id)
);

-- Which embedding model processed this image (NULL = not processed yet).
-- Needed because an image with 0 faces leaves no row in "faces".
ALTER TABLE images ADD COLUMN IF NOT EXISTS faces_model TEXT;

-- One row per detected face. Faces belong to the image *content*, not to a URL.
-- "model" is stored because embeddings from different models are not comparable.
CREATE TABLE IF NOT EXISTS faces (
    id          BIGSERIAL PRIMARY KEY,
    image_id    BIGINT NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    model       TEXT NOT NULL,
    x           INT NOT NULL,
    y           INT NOT NULL,
    w           INT NOT NULL,
    h           INT NOT NULL,
    confidence  REAL NOT NULL,
    embedding   vector(512) NOT NULL
);
CREATE INDEX IF NOT EXISTS faces_image_id_idx ON faces (image_id);
"""


def connect() -> psycopg.Connection:
    conn = psycopg.connect(os.environ["FACES_DB_URL"])
    # Lets psycopg send/receive numpy arrays as pgvector "vector" values
    register_vector(conn)
    return conn


def init_schema() -> None:
    with connect() as conn:
        conn.execute(SCHEMA)


def save_page(conn: psycopg.Connection, page_url: str, image_urls: set[str]) -> None:
    """Upsert a page, its image URLs, and the links between them."""
    # "DO UPDATE ... RETURNING id" returns the id whether the row is new or already existed
    page_id = conn.execute(
        """INSERT INTO pages (url) VALUES (%s)
           ON CONFLICT (url) DO UPDATE SET crawled_at = now()
           RETURNING id""",
        (page_url,),
    ).fetchone()[0]
    for image_url in image_urls:
        image_url_id = conn.execute(
            """INSERT INTO image_urls (url) VALUES (%s)
               ON CONFLICT (url) DO UPDATE SET url = EXCLUDED.url
               RETURNING id""",
            (image_url,),
        ).fetchone()[0]
        conn.execute(
            """INSERT INTO page_image_urls (page_id, image_url_id) VALUES (%s, %s)
               ON CONFLICT DO NOTHING""",
            (page_id, image_url_id),
        )


def image_urls_to_download(conn: psycopg.Connection) -> list[tuple[int, str]]:
    return conn.execute(
        "SELECT id, url FROM image_urls WHERE status IN ('pending', 'failed') ORDER BY id"
    ).fetchall()


def save_image(conn: psycopg.Connection, image_url_id: int, sha256: str,
               file_path: str, width: int, height: int) -> None:
    image_id = conn.execute(
        """INSERT INTO images (sha256, file_path, width, height) VALUES (%s, %s, %s, %s)
           ON CONFLICT (sha256) DO UPDATE SET sha256 = EXCLUDED.sha256
           RETURNING id""",
        (sha256, file_path, width, height),
    ).fetchone()[0]
    conn.execute(
        "UPDATE image_urls SET image_id = %s, status = 'downloaded', error = NULL WHERE id = %s",
        (image_id, image_url_id),
    )


def mark_image_url(conn: psycopg.Connection, image_url_id: int, status: str, error: str) -> None:
    conn.execute(
        "UPDATE image_urls SET status = %s, error = %s WHERE id = %s",
        (status, error[:500], image_url_id),
    )


def images_to_process(conn: psycopg.Connection, model: str) -> list[tuple[int, str]]:
    return conn.execute(
        "SELECT id, file_path FROM images WHERE faces_model IS DISTINCT FROM %s ORDER BY id",
        (model,),
    ).fetchall()


def save_faces(conn: psycopg.Connection, image_id: int, model: str, faces: list) -> None:
    # Delete first so re-processing an image never duplicates its faces
    conn.execute("DELETE FROM faces WHERE image_id = %s AND model = %s", (image_id, model))
    for f in faces:
        conn.execute(
            """INSERT INTO faces (image_id, model, x, y, w, h, confidence, embedding)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
            (image_id, model, f.x, f.y, f.w, f.h, f.confidence, f.embedding),
        )
    conn.execute("UPDATE images SET faces_model = %s WHERE id = %s", (model, image_id))
