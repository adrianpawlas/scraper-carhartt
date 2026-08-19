"""Embedding pipeline using Google SigLIP, run locally via transformers + torch.

`google/siglip-base-patch16-384` is a free public model on the HuggingFace Hub. It
is a dual encoder that produces 768-dim **image** and **text** embeddings. The model
is downloaded once from the Hub and run on the local device (CPU/GPU) — no Inference
API, no tokens, no per-vector network calls.

All returned vectors are L2-normalized and validated to be exactly 768 floats.

Usage:
    embedder = SiglipEmbedder()            # loads the model (blocking)
    img_vec = embedder.generate_image_embedding("https://cdn/...jpg")
    txt_vec = embedder.generate_text_embedding("Title ...")
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
from typing import Optional

import numpy as np
from PIL import Image

from config import CONFIG

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Rate limiter (store request throttling)
# --------------------------------------------------------------------------- #

class RateLimiter:
    """Async rate limiter: allows one call every `interval` seconds."""

    def __init__(self, interval: float) -> None:
        self.interval = interval
        self._lock = asyncio.Lock()
        self._next_slot = 0.0

    async def wait(self) -> None:
        async with self._lock:
            now = asyncio.get_event_loop().time()
            if now < self._next_slot:
                await asyncio.sleep(self._next_slot - now)
            self._next_slot = max(now, self._next_slot) + self.interval


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def l2_normalize(vector: list[float]) -> list[float]:
    arr = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm == 0:
        raise ValueError("cannot normalize a zero vector")
    return (arr / norm).astype(np.float64).tolist()


def _validate(vector: list[float], dim: int) -> bool:
    if not isinstance(vector, list) or len(vector) != dim:
        return False
    for v in vector:
        if not isinstance(v, (int, float)) or not np.isfinite(v):
            return False
    return True


def _resolve_device(pref: str) -> str:
    if pref and pref != "auto":
        return pref
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# --------------------------------------------------------------------------- #
# Local SigLIP embedder (image + text)
# --------------------------------------------------------------------------- #

class SiglipEmbedder:
    """Generates 768-dim image and text embeddings using Google SigLIP.

    The model and processor are loaded once at startup and reused. All methods are
    blocking; call them via ``asyncio.to_thread`` from async code.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        import torch
        from transformers import SiglipModel, SiglipProcessor

        self.model_id = model_id or CONFIG.embedding_model_id
        self.device = _resolve_device(device or CONFIG.device)
        self.embedding_dim = CONFIG.embedding_dim
        self._torch = torch

        log.info("Loading SigLIP model %s on %s", self.model_id, self.device)
        self.processor = SiglipProcessor.from_pretrained(self.model_id)
        self.model = SiglipModel.from_pretrained(self.model_id).to(self.device)
        self.model.eval()

        self.http_client = __import__("httpx").Client(
            timeout=60,
            follow_redirects=True,
            headers={"User-Agent": "Finds-Scraper-Carhatt/1.0"},
        )
        self._dummy_image: Optional[Image.Image] = None
        log.info("SigLIP model loaded (%d dims)", self.embedding_dim)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def generate_image_embedding(self, image_url: str) -> Optional[list[float]]:
        """Embed a single image *image_url* -> 768-dim normalized list (or None)."""
        image = self._download_image(image_url)
        if image is None:
            return None
        try:
            inputs = self.processor(
                images=image,
                text=[" "],  # SigLIP forward() expects both image and text inputs
                return_tensors="pt",
                padding="max_length",
                max_length=2,
            ).to(self.device)
            with self._torch.no_grad():
                emb = self.model(**inputs).image_embeds[0].cpu().tolist()
            return self._finalize(emb, image_url)
        except Exception as exc:  # noqa: BLE001 - embedding failures are non-fatal
            log.error("image embed failed for %s: %s", image_url[:120], exc)
            return None

    def generate_text_embedding(self, text: str) -> Optional[list[float]]:
        """Embed a text string -> 768-dim normalized list (or None)."""
        if not text or not text.strip():
            return None
        try:
            inputs = self.processor(
                text=[text],
                images=self._dummy(),  # SigLIP forward() expects both inputs
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=64,
            ).to(self.device)
            with self._torch.no_grad():
                emb = self.model(**inputs).text_embeds[0].cpu().tolist()
            return self._finalize(emb, text[:120])
        except Exception as exc:  # noqa: BLE001
            log.error("text embed failed: %s", exc)
            return None

    def close(self) -> None:
        self.http_client.close()

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _finalize(self, emb: list[float], label: str) -> Optional[list[float]]:
        if not _validate(emb, self.embedding_dim):
            log.warning("embedding has wrong dim (%d != %d) for %s", len(emb), self.embedding_dim, label)
            return None
        return l2_normalize(emb)

    def _dummy(self) -> Image.Image:
        if self._dummy_image is None:
            self._dummy_image = Image.new("RGB", (384, 384), color=0)
        return self._dummy_image

    def _download_image(self, url: str) -> Optional[Image.Image]:
        for attempt in range(3):
            try:
                resp = self.http_client.get(url)
                resp.raise_for_status()
                img = Image.open(io.BytesIO(resp.content))
                if img.mode != "RGB":
                    img = img.convert("RGB")
                return img
            except Exception as exc:  # noqa: BLE001
                log.warning("image download failed for %s (attempt %d): %s", url[:120], attempt + 1, exc)
                if attempt < 2:
                    time.sleep(1 + attempt)
        return None