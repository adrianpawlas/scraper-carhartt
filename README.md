# Carhartt WIP Scraper (`scraper-carhatt`)

Production-grade scraper for [carhartt-wip.com](https://www.carhartt-wip.com). Discovers every
product, extracts full metadata, generates 768-dim image/text embeddings, and upserts into the
shared Finds Supabase `products` table under `source = "scraper-carhatt"`. Runs on a schedule via
GitHub Actions (never local cron).

## What it does

1. **Discovery** — walks each category page (`?page=N`) until the store's "page does not exist"
   page is reached. Categories: `men`, `women`, `accessories`, `accessories-sale`, `women-sale`.
2. **Parse** — each product page is read from JSON-LD (`application/ld+json`), the Next.js RSC
   payload (`self.__next_f.push`) and rendered HTML (`data-testid` attributes). Captures title,
   description, original price vs. sale price, gender, sizes, SKU / product key, materials,
   care instructions, color, gallery images, etc.
3. **Embed** — generates three 768-dim vectors per product with Google SigLIP
   (`google/siglip-base-patch16-384`):
   - `image_embedding` — front packshot
   - `back_image_embedding` — back view (only when the product has one)
   - `info_embedding` — textual metadata (title, description, price, category, gender, tags, …)
4. **Diff + upsert** — only changed fields and missing embeddings are written; existing rows that
   are identical are skipped (idempotent). Batches use `INSERT ... ON CONFLICT (source, product_url)
   DO UPDATE` via PostgREST `Prefer: resolution=merge-duplicates` (10 rows / batch, 3 retries).
5. **Stale cleanup** — a product not seen for 2 consecutive runs is deleted; a single miss is
   recorded in `metadata.scrape_miss_count` instead.

## Embeddings: local SigLIP (no API)

SigLIP is a **free, public model on the HuggingFace Hub**. The scraper downloads it once and runs
it locally with `transformers` + `torch` — no Inference API, no Gemini, no API keys, no per-vector
network calls. `google/siglip-base-patch16-384` is a dual encoder, so the **same model** produces
both the 768-dim image embeddings and the 768-dim text embeddings.

- Runs on `DEVICE=auto` (CUDA → MPS → CPU); override with `DEVICE=cpu` etc.
- All vectors are L2-normalized to unit length (matches the other Finds scrapers).
- On GitHub Actions the model is cached at `~/.cache/huggingface` so scheduled runs don't
  re-download it.

## Back-view detection

Back-view detection is based on the **gallery image alt text** in the RSC payload: images are
classified as "from the front" / "from the back". The `ST-01`/`ST-02` filename suffix is **not
reliable** (e.g. the Oakland Pant ships `ST-01` = back, `ST-02` = front) and is only used to pick
the best quality image once the view has been classified.

## Requirements

- Python 3.11+
- Dependencies in `requirements.txt` (`httpx`, `beautifulsoup4`, `Pillow`, `numpy`,
  `python-dotenv`, plus `torch`, `transformers`, `sentencepiece`, `protobuf` for SigLIP)

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # fill in SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
```

`HF_TOKEN` is optional — only needed if your network/proxy requires authentication to download
models from the Hub.

## Usage

```bash
python main.py                    # scrape everything, upsert, embed what's missing
python main.py --category accessories-sale --limit 10   # small test run
python main.py --dry-run         # discovery only (no product fetch / embeddings / DB writes)
python main.py --no-embed        # write rows but skip embeddings (fast seeding)
```

A summary is printed at the end; JSON logs go to stderr. Products that fail to upsert are appended
to `failed_products.log`.

## Env vars

| Variable | Required | Purpose |
| --- | --- | --- |
| `SUPABASE_URL` | yes | Supabase project URL |
| `SUPABASE_SERVICE_ROLE_KEY` | yes | service-role key (reads/writes `public.products`) |
| `HF_TOKEN` | no | optional, for authenticating model downloads from the Hub |
| `DEVICE` | no | `auto` (default) / `cpu` / `cuda` / `mps` |
| `EMBEDDING_MODEL_ID` | no | model override (default `google/siglip-base-patch16-384`) |
| `SCRAPE_LIMIT` | no | cap on products processed per run |
| `REQUEST_DELAY_SECONDS` | no | delay between store requests (default 0.6) |
| `EMBEDDING_DELAY_SECONDS` | no | delay between embedding calls (default 0.1) |
| `FETCH_CONCURRENCY` | no | product page fetch concurrency (default 4) |
| `DRY_RUN` | no | discovery-only mode |

## Supabase schema notes

The real `products` table schema differs from the original task spec and the client is
schema-aware:

- There is **no** `embedding_version` column and **no** `last_seen_at` column. Front/back
  embedding presence is tracked via `metadata.embedding_flags = {"image": true, "back": true}`.
  `last_seen_at`-style bookkeeping lives in `metadata` (`scraped_at`, `last_seen_at`,
  `scrape_miss_count`, `last_missed_at`).
- `image_url` is `NOT NULL`.
- Vectors are 768-dim (`vector(768)`) and are written as JSON arrays (PostgREST accepts
  `[1.0, 2.0, ...]` for a `vector` column).
- Extra columns exist (`blur_hash`, `brand_tsv`, `currency`, `dominant_color`, `feed_image_url`,
  `image_height`, `image_width`, `likes_count`, `saves_count`, `search_tsv`, `search_vector`,
  `shares_count`, `title_tsv`, `description_tsv`, ...) — the scraper only writes columns it knows
  about and leaves the rest untouched.

## GitHub Actions

The workflow `.github/workflows/scrape.yml` runs the scraper on Sunday and Thursday at 04:45 UTC
(`workflow_dispatch` also available, with optional `limit` and `skip_embeddings` inputs) and
uploads `failed_products.log` as an artifact when present. It installs `libgl1 libglib2.0-0`
(Pillow deps), caches the SigLIP model, and runs with `DEVICE=cpu`.

Set the following repository secrets:

- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `HF_TOKEN` (optional)

## Source identity

Rows are written with `source = "scraper-carhatt"` and `id = sha256(source + ":" + product_url)[:32]`,
the upsert conflict key is `(source, product_url)`. Keep the source value stable — changing it
would create duplicate product sets.