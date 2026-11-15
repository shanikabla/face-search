"""Find the pages showing the same person as a query face (cosine distance in pgvector)."""
from dataclasses import dataclass

import numpy as np
import psycopg

# Max cosine distance to call two faces "the same person". Tuned on our data: see README.
DEFAULT_MAX_DISTANCE = 0.30

# Exact nearest-neighbour scan: with a few thousand faces it takes milliseconds and
# never misses a match (an HNSW index would be approximate; see README for scaling).
# "<=>" is pgvector's cosine distance: 0 = same direction, 1 = unrelated, 2 = opposite.
MATCHING_PAGES_SQL = """
WITH matching_faces AS (
    SELECT f.image_id, f.embedding <=> %(query)s AS distance
    FROM faces f
    WHERE f.model = %(model)s                    -- never compare across models
      AND f.embedding <=> %(query)s < %(max_distance)s
)
SELECT p.url                                       AS page_url,
       min(m.distance)                             AS best_distance,
       count(DISTINCT m.image_id)                  AS matching_images,
       array_agg(DISTINCT iu.url ORDER BY iu.url)  AS image_urls
FROM matching_faces m
JOIN image_urls iu       ON iu.image_id = m.image_id
JOIN page_image_urls piu ON piu.image_url_id = iu.id
JOIN pages p             ON p.id = piu.page_id
GROUP BY p.url
ORDER BY best_distance
"""


@dataclass
class PageMatch:
    page_url: str
    best_distance: float
    matching_images: int
    image_urls: list[str]

    @property
    def similarity(self) -> float:
        return 1.0 - self.best_distance


def find_matching_pages(conn: psycopg.Connection, embedding: np.ndarray, model: str,
                        max_distance: float = DEFAULT_MAX_DISTANCE) -> list[PageMatch]:
    rows = conn.execute(
        MATCHING_PAGES_SQL,
        {"query": embedding, "model": model, "max_distance": max_distance},
    ).fetchall()
    return [PageMatch(url, float(dist), n, urls) for url, dist, n, urls in rows]
