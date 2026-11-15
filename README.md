# Face Search

Crawls websites, finds the people in their images, and indexes every face as an
embedding vector. Then, given a photo of someone, `face_matcher.py` returns the pages of the
crawled sites that show the same person.

```
                       Airflow DAG "face_pipeline"
┌──────────────┐   ┌──────────────────┐   ┌─────────────────┐   ┌──────────────┐
│ init_schema  │──▶│ crawl_site (×N)  │──▶│ download_images │──▶│ detect_faces │
│ tables (idem-│   │ one task per site│   │ JPEG, ≤1024px,  │   │ RetinaFace + │
│ potent)      │   │ in parallel      │   │ dedup by SHA-256│   │ Facenet512   │
└──────────────┘   └──────────────────┘   └─────────────────┘   └──────┬───────┘
                                                                       ▼
                                                     PostgreSQL + pgvector ("faces" DB)
                                                                       ▲
         face_matcher.py: query photo ─▶ same detector + model ─▶ cosine search ─▶ pages
```

Everything runs in Docker Compose: PostgreSQL 16 with pgvector, Airflow 3 (API server,
scheduler, DAG processor), and the search CLI.

---

## 1. How to run the pipeline

**Requirements:** Docker with Compose v2, about 8 GB of RAM for Docker, about 8 GB of disk
space (the image includes TensorFlow).

```bash
git clone <this repo> face-search && cd face-search
cp .env.example .env          # then replace each "change-me" (e.g. openssl rand -hex 32)
                              # on Linux, also set AIRFLOW_UID=$(id -u)
docker compose up -d --build  # first build takes 5-10 min (TensorFlow, DeepFace)
```

Open the Airflow UI at **http://localhost:8080** (no login: local development setup).
DAGs are paused when created. Unpause `face_pipeline`, then click **Trigger** and edit the
parameters in the form if needed. The same thing from the command line:

```bash
docker compose exec airflow-scheduler airflow dags unpause face_pipeline
docker compose exec airflow-scheduler airflow dags trigger face_pipeline -c '{"max_pages": 10}'
```

| Parameter | Default | Meaning |
|---|---|---|
| `start_urls` | `["https://edition.cnn.com/", "https://people.com/"]` | Sites to crawl, one parallel task each |
| `max_pages` | `50` | Max HTML pages crawled **per site** |
| `crawl_delay_s` | `0.5` | Pause between two requests to the same site |
| `max_image_size` | `1024` | Longest side of the normalized images, in pixels |

**Run time:** crawling and downloading are fast. Face detection is the slow part, at about
**1 s per image on CPU**. With `max_pages: 10`, expect a few minutes. With the defaults,
several thousand images and about an hour.

The pipeline is **idempotent and resumable**: re-running it never creates duplicates, and an
interrupted run continues where it stopped (each page, image and face set is committed
on its own). Images that are already processed are skipped.

Normalized images are written to `data/images/<sha256>.jpg`. To inspect the database:

```bash
docker compose exec postgres psql -U faces faces
```

---

## 2. How to use the face matcher

Put a photo in `queries/` (the folder is mounted in the container), then:

```bash
docker compose run --rm face-matcher queries/person.jpg
```

| Option | Effect |
|---|---|
| `--max-distance 0.25` | Stricter (fewer, surer matches) or looser threshold. Default: `0.30` |
| `--all-faces` | Search every face of the photo. By default, only the largest (main) face |
| `--json` | Machine-readable output |

The pipeline must have run at least once. The script exits with an error if the index is
empty, if the file doesn't exist, or if no face is found in the photo.

### Example and expected output

Download a query photo (any photo of the person works; this one is from Wikipedia, so it is
**not** one of the crawled images):

```bash
curl -sL -A "face-search-example" -o queries/travis_barker.jpg \
  "https://upload.wikimedia.org/wikipedia/commons/a/af/Travis_Barker_2019_by_Glenn_Francis_%28tighter_portrait_crop%29.jpg"
docker compose run --rm face-matcher queries/travis_barker.jpg
```

Output on our crawl of 2026-11-15 (image lists shortened):

```
Query face at x=276 y=142 (408x590px), max distance 0.3
  Found on 2 page(s):

  similarity 0.915  https://people.com/travis-barker-cutest-family-photos-kids-stepchildren-12253429
      image: https://people.com/thmb/.../Travis-Barker-Kids-8-111326-ebb6fa670cbe4527a7dea188aa977bd8.jpg
      image: https://people.com/thmb/.../travis-barker-4-045fbaf8fbb74da3bd01656e16145d1b.jpg
      ... (8 images on this page)
  similarity 0.730  https://people.com/kathy-ireland-reveals-husband-was-rescued-from-burning-boat-after-engine-exploded-12255338
      image: https://people.com/thmb/.../Kourtney-Kardashian-Travis-Barker-111425-2228700a2bfd4980bd679322abf36789.jpg
```

The second page is about someone else, but shows a "related article" thumbnail of Travis
Barker. That is exactly what the search should find. **Similarity** is `1 - cosine distance`,
and pages are sorted by their best matching face.

A person who is not on the sites (negative control, Marie Curie):

```
Query face at x=447 y=422 (562x737px), max distance 0.3
  No page found with this person.
```

With `--json`:

```json
[
  {
    "query_face": {"x": 276, "y": 142, "w": 408, "h": 590, "confidence": 1.0},
    "pages": [
      {
        "page_url": "https://people.com/travis-barker-cutest-family-photos-kids-stepchildren-12253429",
        "similarity": 0.915,
        "distance": 0.085,
        "image_urls": ["https://people.com/thmb/.../Travis-Barker-Kids-8-...jpg", "..."]
      }
    ]
  }
]
```

The sites change every day, so your own crawl will index different articles and people.

---

## 3. Libraries and models

| Purpose | Library / model |
|---|---|
| Orchestration | Apache Airflow 3.1 (LocalExecutor) |
| Database + vector search | PostgreSQL 16 + pgvector, psycopg 3 |
| Crawling | requests, BeautifulSoup 4, `urllib.robotparser` |
| Image normalization | Pillow |
| Face detection | **RetinaFace** (through DeepFace 0.0.93) |
| Face embeddings | **Facenet512** (through DeepFace, TensorFlow 2.17) |
| Tests | pytest |

Model weights are downloaded by DeepFace on first use and cached in a Docker volume.

---

## 4. Approach and architecture decisions

### Infrastructure

- **LocalExecutor**: everything runs on one machine and the volume is moderate. To scale,
  CeleryExecutor (several workers) or KubernetesExecutor (e.g. GPU nodes only for
  `detect_faces`) would only need configuration changes, not DAG changes.
- **One PostgreSQL instance, two databases**: `airflow` for Airflow's metadata, `faces`
  for our data, with its own user. Separate permissions and lifecycles (the face index
  can be reset without touching Airflow's history), and one service less than a dedicated
  vector database.
- **pgvector instead of a separate vector DB** (Qdrant, Milvus): metadata, bounding
  boxes and embeddings live together, and the search is one SQL query with joins.
- **Secrets in `.env`** (git-ignored), with `.env.example` committed. Compose fails
  immediately with a clear message if a variable is missing (`${VAR:?...}`).
- **Business logic outside the DAG**: `face_search/` has no Airflow dependency. It is
  testable on its own and shared between the DAG and `face_matcher.py`. The DAG only
  chains the steps.

### Data model

```
pages ──< page_image_urls >── image_urls ──> images ──< faces
(url UNIQUE)                  (url UNIQUE)   (sha256 UNIQUE)  (bbox, confidence,
"where it was seen"                          "the content"     model, vector(512))
```

- **Two levels of deduplication**: the same image URL on several pages is stored once
  (unique normalized absolute URL), but every page↔image link is kept, otherwise the
  search could only return one page. The same file behind different URLs is stored once
  (SHA-256 of the downloaded bytes).
- **Faces belong to the image content**, not to a URL: a face is computed once, however
  many pages show it.
- **`faces.model`**: embeddings from two models are not comparable, even with the same
  dimension. Mixing them would give a silent bug (valid SQL, meaningless distances), so the
  model is stored and every search filters on it.
- **`images.faces_model`** records which model processed an image, because an image with
  no face leaves no row in `faces` and would otherwise be re-processed on every run.
- **Schema created by the pipeline** (`CREATE TABLE IF NOT EXISTS`, `ADD COLUMN IF NOT
  EXISTS`) rather than by the database init script, which only runs on an empty volume.

### Crawling

- **Breadth-first**, same site only (`www.` ignored), respects `robots.txt`, with a
  delay between requests and a page limit.
- **One task per site** with Airflow dynamic task mapping
  (`crawl_site.expand(start_url=...)`). Sites crawl in parallel, and each has its own
  status, logs and retries. The list goes through a small task because params are only
  known at run time, while the DAG file is parsed long before.
- **HTML is parsed (BeautifulSoup), not regex-matched for `.jpg`**: real image URLs often
  have no extension (CDNs), carry query strings, are `.webp`, relative, or lazy-loaded.
  Images are collected from `src`, `srcset` (largest candidate), `data-src` and similar
  lazy-loading attributes, `<picture><source>` and `og:image`.
- **Non-HTML links are skipped**: `.pdf`, `.zip` and `.mp4` paths are filtered out before any
  request, and pages are fetched with `stream=True` so that a non-HTML body is never
  downloaded. The `Content-Type` check stays the source of truth.

### Image normalization

Every image is checked (`Content-Type: image/*`, then opened by Pillow), then:
EXIF orientation applied (phone photos are often stored sideways, and a rotated face is
missed by the detector), transparency painted white (JPEG has no alpha, and a naive
conversion turns it black), converted to RGB JPEG, and shrunk to at most 1024 px on the
longest side (aspect ratio kept, never upscaled). Images under 64 px are rejected as
icons.

Failures are classified as **`rejected`** when retrying can't help (not an image, HTTP
4xx, too small) or **`failed`** when it can (timeout, HTTP 5xx, 429). Only `failed`
images are retried on the next run.

### Face detection: benchmarked on our own images

The 4 detectors of DeepFace on 40 random images from the index
(`scripts/benchmark_detectors.py`):

| Detector | Faces | Images with ≥ 1 face | Faces < 40 px | Speed |
|---|---|---|---|---|
| OpenCV (Haar) | 30 | 22/40 | 3 | 0.04 s/img |
| YuNet | 28 | 20/40 | 1 | 0.02 s/img |
| MTCNN | 36 | 24/40 | 6 | 0.13 s/img |
| **RetinaFace** | **47** | **27/40** | 8 | 0.78 s/img |

Looking at the images where they disagree: only RetinaFace found profile faces and the
faces in an artistic double exposure. It is 40× slower than YuNet, but this is a **batch**
job whose cost is paid once, and the quality of the index is what matters. A real-time
use case (< 33 ms per frame) would call for YuNet. A search processes a single image, so
it stays fast.

**Filters** applied after detection:

- **Confidence ≥ 0.98.** A contact sheet of the least confident detections between 0.9 and
  0.97 showed a puma, hands holding a camera and a carved ornament. 0.98 removes them
  for only 2% of the faces.
- **Smallest side ≥ 40 px.** Smaller faces become a blur once resized for the model. Their
  embeddings land close to every other blur, near the "average face", so they don't
  just add nothing: they show up as **false matches** for unrelated queries.

**Alignment** (`align=True`): faces are rotated upright using the eyes before embedding,
because the models were trained on upright faces.

### Embedding model and similarity threshold

**Facenet512** (512 dimensions) ranks first among DeepFace's models in the LFW
benchmark published by DeepFace. The search uses **cosine distance** (`<=>` in pgvector).

**How the threshold was chosen.** `scripts/nearest_faces.py` lists the 12 nearest indexed
faces of a query, whatever their distance, with a contact sheet. I ran it on 12 Wikipedia
portraits (photos that are **not** in the index): 9 people who appear in the crawled articles,
1 I couldn't verify, and 2 **negative controls** who appear nowhere (Marie Curie, Alan Turing).
I then labelled each neighbour by eye:

| Case | Distances of true matches | Example |
|---|---|---|
| Frontal photo, similar age | **0.085 – 0.34** | Travis Barker, Meghan Markle, Rihanna, Trump, Kathy Ireland, Rebecca Gayheart |
| Large age gap between the photos | 0.37 – 0.46 | Sade (Wikipedia photo ~30 years after the ones on the site) |
| Profile or low-resolution faces on the site | 0.40 – 0.52 | Joe Rogan, Luke Grimes |

| First **impostors** (another person) | Distance |
|---|---|
| Another woman, for Rihanna | 0.357 |
| Paul McCartney, for Marie Curie | 0.376 |
| Tom Hanks, for Joe Rogan | 0.390 |
| Nearest face to Alan Turing | 0.401 |

**Default: 0.30.** The assignment asks for matches "with high confidence", so precision
comes first: below 0.30 I found no clear impostor, and the first one is at 0.357, which
leaves a margin. It is also the value DeepFace publishes for Facenet512 with cosine
distance, and our own data confirms it. The cost: hard cases (big age gap, profile shots)
are missed. `--max-distance 0.35` finds more for frontal photos while staying under the
first impostor seen. Above 0.37, impostors start to appear.

One borderline case is worth knowing about: at 0.23 from Meghan Markle, the index has a face
from the article about her half-siblings. Family resemblance is a known weakness of face
recognition.

### Search

`face_matcher.py` detects the face in the query photo with **the same detector, filters and
model** as the pipeline, then runs a single SQL query: faces under the threshold → images →
image URLs → pages, grouped by page and sorted by best distance. It is an **exact** scan
(milliseconds for thousands of faces, never misses a match). For millions of faces, an
HNSW index (`CREATE INDEX ... USING hnsw (embedding vector_cosine_ops)`) would make it
approximate but logarithmic.

### Results on our crawl (2026-11-15, `max_pages: 30`)

| | |
|---|---|
| Pages crawled | 67 (32 CNN, 35 People) |
| Images downloaded | 4,374 URLs → **4,335 distinct images** (SHA-256 dedup) |
| Rejected / failed | 22 rejected (SVG, icons, 4xx), 1 failed (network) |
| Images with ≥ 1 face | **2,315** (53%) |
| Indexed faces | **3,319** (smallest side 40-606 px, 86 px on average) |
| Pages with ≥ 1 face | 63 |
| Face detection speed | ~1 s per image on CPU (Apple Silicon, 12 cores, in Docker) |
| Search time | ~7 s, almost all of it loading TensorFlow; the SQL query takes milliseconds |

---

## 5. Tests

```bash
docker compose run --rm --entrypoint pytest face-matcher
```

Unit tests cover the pure functions where the real bugs were found: `srcset` parsing (CNN
puts commas **inside** its image URLs, which first broke a naive split on `,`), link
filtering, image URL extraction, and image normalization (resize, transparency, EXIF
rotation, rejection of icons and non-images).

Analysis scripts used for the decisions above:

```bash
docker compose run --rm --entrypoint python face-matcher scripts/benchmark_detectors.py
docker compose run --rm --entrypoint python face-matcher scripts/nearest_faces.py queries/*.jpg
```

---

## 6. Limitations and possible improvements

- **Same photo in several sizes** (CNN serves `w_256`, `w_480`, `w_780` variants): the bytes differ,
  so SHA-256 doesn't deduplicate them, and their faces are embedded more than once. A
  perceptual hash would merge them. The impact is limited to duplicate computation:
  results are grouped by page, so the search output is unaffected.
- **Statues, paintings and animals** can pass the detector (a statue *is* a face,
  geometrically). They rarely match a real person's query.
- **Profile and low-resolution faces** produce less reliable embeddings: true matches can
  fall above the threshold (see the Joe Rogan example).
- **JavaScript-rendered sites** would need a headless browser (Playwright). CNN and People
  are server-rendered.
- **Small CDN thumbnails**: requesting a larger variant (rewriting `w_256` in the URL) would
  give larger faces and better embeddings.
- **Scale**: CeleryExecutor or KubernetesExecutor with GPU workers for `detect_faces`, an HNSW index,
  and batching faces through the model instead of one image at a time.
- **Production hardening**: real authentication on the Airflow UI, secrets from a secrets
  manager, code baked into the image instead of bind-mounted.

---

## Project structure

```
├── dags/face_pipeline.py     Airflow DAG: only chains the steps
├── face_search/              Business logic (no Airflow dependency)
│   ├── crawler.py            BFS crawl, image URL and link extraction
│   ├── images.py             Download, validation, normalization
│   ├── faces.py              Detection + embeddings (shared with the search)
│   ├── search.py             Cosine-distance search grouped by page
│   └── db.py                 Schema and queries
├── face_matcher.py           Search CLI
├── scripts/                  Detector benchmark, threshold calibration
├── tests/                    pytest unit tests
├── db/init.sh                Creates the "faces" DB, its user and pgvector
├── queries/                  Put query photos here (not committed)
├── docker-compose.yml, Dockerfile, requirements.txt, .env.example
```
