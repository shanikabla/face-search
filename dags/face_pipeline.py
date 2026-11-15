"""Crawl sites, download and normalize their images, detect faces and store embeddings."""
import logging
import os
from pathlib import Path

import requests
from airflow.sdk import Param, dag, get_current_context, task

from face_search import crawler, db, images

log = logging.getLogger(__name__)
IMAGES_DIR = Path(os.environ.get("DATA_DIR", "/opt/airflow/data")) / "images"


@dag(
    schedule=None,
    catchup=False,
    tags=["face-search"],
    params={
        # Defaults can be overridden per run (Trigger form in the UI, or -c on the CLI)
        "start_urls": Param(
            ["https://edition.cnn.com/", "https://people.com/"],
            type="array", items={"type": "string"}, minItems=1,
        ),
        "max_pages": Param(50, type="integer", minimum=1),  # per site
        "crawl_delay_s": Param(0.5, type="number", minimum=0),
        "max_image_size": Param(1024, type="integer", minimum=64),
    },
)
def face_pipeline():
    @task
    def init_schema():
        db.init_schema()

    @task
    def list_start_urls() -> list[str]:
        # Params are only known at run time, so expand() needs them via a task's output
        return get_current_context()["params"]["start_urls"]

    @task(map_index_template="{{ task.op_kwargs['start_url'] }}")
    def crawl_site(start_url: str):
        params = get_current_context()["params"]
        session = crawler.make_session()
        pages = 0
        with db.connect() as conn:
            for page_url, image_urls in crawler.crawl(
                start_url, params["max_pages"], params["crawl_delay_s"], session
            ):
                db.save_page(conn, page_url, image_urls)
                conn.commit()  # one page at a time: a crash keeps what's already crawled
                pages += 1
        log.info("Crawled %d pages from %s", pages, start_url)

    @task
    def download_images():
        params = get_current_context()["params"]
        session = crawler.make_session()
        counts = {"downloaded": 0, "rejected": 0, "failed": 0}
        with db.connect() as conn:
            for image_url_id, url in db.image_urls_to_download(conn):
                try:
                    data = images.download(url, session)
                    img = images.normalize(data, IMAGES_DIR, params["max_image_size"])
                    db.save_image(conn, image_url_id, img.sha256, img.file_path, img.width, img.height)
                    counts["downloaded"] += 1
                except images.ImageRejected as exc:
                    db.mark_image_url(conn, image_url_id, "rejected", str(exc))
                    counts["rejected"] += 1
                except requests.RequestException as exc:
                    db.mark_image_url(conn, image_url_id, "failed", str(exc))
                    counts["failed"] += 1
                conn.commit()
        log.info("Images: %s", counts)

    @task
    def detect_faces():
        # Imported here, not at the top: loading TensorFlow takes seconds and
        # would slow down the dag-processor, which parses this file constantly
        from face_search import faces

        counts = {"images": 0, "with_faces": 0, "faces": 0, "errors": 0}
        with db.connect() as conn:
            for image_id, file_path in db.images_to_process(conn, faces.MODEL_NAME):
                try:
                    found = faces.detect_and_embed(file_path)
                except Exception:
                    # Not marked as processed: it will be retried on the next run
                    log.exception("Face detection failed for image %s (%s)", image_id, file_path)
                    counts["errors"] += 1
                    continue
                db.save_faces(conn, image_id, faces.MODEL_NAME, found)
                conn.commit()
                counts["images"] += 1
                counts["with_faces"] += bool(found)
                counts["faces"] += len(found)
        log.info("Faces: %s", counts)

    # One crawl_site task instance per site, running in parallel
    crawled = crawl_site.expand(start_url=list_start_urls())
    init_schema() >> crawled >> download_images() >> detect_faces()


face_pipeline()
