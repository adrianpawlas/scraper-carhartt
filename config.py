"""Environment-based configuration for the Carhartt WIP scraper.

All secrets are read from the environment (GitHub Actions secrets or .env).
No secret values are hardcoded here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _get(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


@dataclass(frozen=True)
class Config:
    # --- Brand / source identity ---
    brand: str = "Carhartt"
    source: str = "scraper-carhatt"

    # --- Store ---
    landing_url: str = "https://www.carhartt-wip.com/en-bg"
    locale: str = "en-bg"
    country: str = "BG"
    category_urls: tuple = (
        "https://www.carhartt-wip.com/en-bg/c/men",
        "https://www.carhartt-wip.com/en-bg/c/women",
        "https://www.carhartt-wip.com/en-bg/c/accessories",
        "https://www.carhartt-wip.com/en-bg/c/accessories-sale",
        "https://www.carhartt-wip.com/en-bg/c/women-sale",
    )
    user_agent: str = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
    )

    # --- Supabase ---
    supabase_url: str = field(default_factory=lambda: _get("SUPABASE_URL", "").rstrip("/"))
    supabase_service_role_key: str = field(default_factory=lambda: _get("SUPABASE_SERVICE_ROLE_KEY"))

    # --- Embeddings (local SigLIP via transformers/torch, free HF Hub model) ---
    embedding_model_id: str = field(default_factory=lambda: _get("EMBEDDING_MODEL_ID", "google/siglip-base-patch16-384"))
    embedding_dim: int = 768
    embedding_version: int = 2
    device: str = field(default_factory=lambda: _get("DEVICE", "auto"))  # auto|cpu|cuda|mps
    embedding_delay: float = field(default_factory=lambda: float(_get("EMBEDDING_DELAY_SECONDS", "0.1") or 0.1))

    # --- Behaviour ---
    scrape_limit: int = field(default_factory=lambda: int(_get("SCRAPE_LIMIT", "0") or 0))
    request_delay: float = field(default_factory=lambda: float(_get("REQUEST_DELAY_SECONDS", "0.6") or 0.6))
    fetch_concurrency: int = field(default_factory=lambda: int(_get("FETCH_CONCURRENCY", "4") or 4))
    dry_run: bool = field(
        default_factory=lambda: _get("DRY_RUN", "false").strip().lower() in ("1", "true", "yes")
    )
    max_pages_per_category: int = 200

    @property
    def rest_url(self) -> str:
        return f"{self.supabase_url}/rest/v1/products"

    def require_supabase(self) -> None:
        if not self.supabase_url or not self.supabase_service_role_key:
            raise RuntimeError(
                "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set "
                "(see .env.example and GitHub Actions secrets)."
            )


CONFIG = Config()