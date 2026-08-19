"""Carhartt WIP store parser.

Parses category listing pages (pagination + product cards) and product detail
pages (JSON-LD structured data, Next.js RSC gallery payload and rendered HTML).

Back-view detection rule (documented in README):

  The PDP gallery is embedded in the Next.js RSC payload as a JSON array of
  {key, alt, src}. The FRONT packshot is the first image whose alt text contains
  "from the front" (studio "ST" shots preferred over on-figure "OF" shots); the
  BACK view is the first image whose alt text contains "from the back". The key
  suffix (ST-01 / ST-02) is NOT reliable — on some products ST-01 is the back
  view and ST-02 the front, so detection always relies on the alt text.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional
from urllib.parse import unquote, urlparse

from bs4 import BeautifulSoup

# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

CURRENCY_MAP = {"\u20ac": "EUR", "$": "USD", "\u00a3": "GBP", "z\u0142": "PLN", "K\u010d": "CZK"}


def humanize_category(slug: str) -> str:
    """Turn a category slug into a readable label, splitting combined labels.

    Examples: "men" -> "Men", "women-sale" -> "Women Sale",
              "sweats-hoodies" -> "Sweats, Hoodies"
    """
    label = slug.replace("-", " ").replace(" & ", " & ").strip()
    words = [w for w in label.split(" ") if w]
    if len(words) > 1:
        label = ", ".join(words)
    return label.title()


def gender_from_category(slug: str) -> Optional[str]:
    """Best-effort gender derived from the top-level category slug."""
    s = slug.lower()
    if s.startswith("men") or s.startswith("homme") or s.startswith("herren"):
        return "men"
    if s.startswith("women") or s.startswith("femme") or s.startswith("dame"):
        return "women"
    return None


def normalize_image_url(url: str) -> str:
    """Canonicalize an Amplience image URL.

    Drops width/quality/preset query noise, keeps the ST/OF transform preset and
    fmt=auto so the stored URL is stable across category page and PDP variants.
    """
    url = unquote(url)
    parsed = urlparse(url)
    path = parsed.path
    query = parsed.query
    q_params: list[str] = []
    for part in query.split("&"):
        if not part:
            continue
        key = part.split("=", 1)[0]
        if key in ("$ST$", "$ST$=", "$OF$", "$OF$=", "$3D$", "$MV$",
                   "$google_struc_main_st$", "$google_struc_main_of$"):
            q_params.append(key)
        elif key in ("fmt",):
            q_params.append(part)
    if "fmt=auto" not in q_params:
        q_params.append("fmt=auto")
    return f"https://cdn.media.amplience.net{path}?" + "&".join(q_params)


def compressed_image_url(url: str) -> Optional[str]:
    """Produce an optimized CDN variant (max 1200px, q80) for the same image."""
    base = normalize_image_url(url)
    if "?" not in base:
        return None
    path, query = base.split("?", 1)
    parts = [p for p in query.split("&") if not p.startswith("w=") and not p.startswith("qlt=")]
    parts.append("w=1200")
    parts.append("qlt=80")
    return f"{path}?" + "&".join(parts)


def extract_product_key(image_url: str) -> Optional[str]:
    """Extract the catalog product key (e.g. I036729_3DE_XX) from an image URL."""
    m = re.search(r"/carhartt_wip/([^/?]+)", image_url)
    if not m:
        return None
    asset = m.group(1)
    asset = re.sub(r"-(?:ST|OF|MV|3D)-\d+$", "", asset)
    return asset or None


def format_price(raw: Optional[str], fallback_value: Optional[float], fallback_currency: str = "EUR") -> Optional[str]:
    """Convert a store price string like '27,30 €' into '27.30EUR'."""
    if raw:
        m = re.search(r"(\d[\d.,]*)\s*([\u20ac$£złKč])", raw)
        if m:
            try:
                value = float(m.group(1).replace(".", "").replace(",", "."))
            except ValueError:
                value = None
            if value is not None:
                code = CURRENCY_MAP.get(m.group(2), fallback_currency)
                return f"{value:.2f}{code}"
    if fallback_value is not None:
        return f"{float(fallback_value):.2f}{fallback_currency}"
    return None


def clean_text(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


# --------------------------------------------------------------------------- #
# RSC payload helpers
# --------------------------------------------------------------------------- #

def extract_rsc_text(html: str) -> str:
    """Concatenate and decode the Next.js RSC flight payload chunks."""
    parts: list[str] = []
    for m in re.finditer(r"<script>self\.__next_f\.push\(\[1,\"(.*?)\"\]\)</script>", html, re.S):
        try:
            parts.append(m.group(1).encode("utf-8", "surrogatepass").decode("unicode_escape"))
        except Exception:
            continue
    return "".join(parts)


def extract_gallery_images(rsc_text: str) -> list[dict[str, Any]]:
    """Extract the PDP gallery images array from the RSC payload."""
    i = rsc_text.find('"pdp-image-gallery"')
    if i == -1:
        return []
    j = rsc_text.find('"images":', i)
    if j == -1:
        return []
    start = rsc_text.find("[", j)
    if start == -1:
        return []
    depth = 0
    k = start
    while k < len(rsc_text):
        if rsc_text[k] == "[":
            depth += 1
        elif rsc_text[k] == "]":
            depth -= 1
            if depth == 0:
                break
        k += 1
    raw = rsc_text[start : k + 1]
    try:
        images = json.loads(raw)
    except Exception:
        return []
    if not isinstance(images, list):
        return []
    out = []
    for im in images:
        if isinstance(im, dict) and isinstance(im.get("src"), str):
            out.append(
                {
                    "key": im.get("key"),
                    "alt": (im.get("alt") or "").lower(),
                    "src": im.get("src"),
                    "type": im.get("type"),
                }
            )
    return out


# --------------------------------------------------------------------------- #
# Category pages
# --------------------------------------------------------------------------- #

CATEGORY_CARD_SELECTOR = "article[data-testid='product-card']"


def parse_category_page(html: str) -> list[dict[str, Any]]:
    """Parse product cards on a category listing page."""
    soup = BeautifulSoup(html, "html.parser")
    products: list[dict[str, Any]] = []
    for card in soup.select(CATEGORY_CARD_SELECTOR):
        link = card.select_one("a[href*='/p/']")
        if not link:
            continue
        product_url = link.get("href")
        if not product_url or not product_url.startswith("http"):
            base = "https://www.carhartt-wip.com"
            product_url = base + product_url
        title = card.select_one("[data-testid='product-card-title']")
        desc = card.select_one("[data-testid='product-card-descriptions']")
        price_el = card.select_one("[data-testid='price']")
        pre_price_el = card.select_one("[data-testid='pre-sale-price']")
        color = None
        if desc:
            parts = desc.get_text(" ", strip=True).split(" ", 1)
            color = clean_text(parts[1] if len(parts) > 1 else parts[0])

        front_image = back_image = None
        for img in card.select("img"):
            alt = (img.get("alt") or "").lower()
            src = img.get("src") or img.get("data-src")
            if not src:
                continue
            if "from the front" in alt and front_image is None:
                front_image = normalize_image_url(src)
            elif "from the back" in alt and back_image is None:
                back_image = normalize_image_url(src)

        products.append(
            {
                "product_url": product_url,
                "title": clean_text(title.get_text(" ", strip=True)) if title else None,
                "color": color,
                "price": format_price(price_el.get_text(" ", strip=True) if price_el else None, None),
                "pre_sale_price": (
                    format_price(pre_price_el.get_text(" ", strip=True) if pre_price_el else None, None)
                ),
                "front_image": front_image,
                "back_image": back_image,
            }
        )
    return products


def is_not_found_page(html: str) -> bool:
    """Detect the store's 'page does not exist' page (pagination termination)."""
    return (
        "data-testid=\"product-card\"" not in html
        and ("doesn\u2019t exist" in html or "doesn't exist" in html or "doesn&rsquo;t exist" in html)
    )


# --------------------------------------------------------------------------- #
# Product detail page
# --------------------------------------------------------------------------- #

def parse_product_jsonld(soup: BeautifulSoup) -> dict[str, Any]:
    for script in soup.find_all("script", {"type": "application/ld+json"}):
        try:
            data = json.loads(script.string or script.get_text())
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        if data.get("@type") in ("Product",):
            return data
    return {}


def _gallery_pick(gallery: list[dict[str, Any]], marker: str) -> Optional[dict[str, Any]]:
    """Pick the preferred image: studio 'ST' first, then any match."""
    candidates = [im for im in gallery if marker in im["alt"]]
    if not candidates:
        return None
    for im in candidates:
        if str(im.get("type")) == "ST":
            return im
    return candidates[0]


def parse_product_page(html: str, rsc_text: str) -> dict[str, Any]:
    """Extract every field of a product detail page."""
    soup = BeautifulSoup(html, "html.parser")
    ld = parse_product_jsonld(soup)

    title_el = soup.select_one("[data-testid='product-title']")
    title = clean_text(title_el.get_text(" ", strip=True)) if title_el else None
    if not title and ld.get("name"):
        title = clean_text(str(ld["name"]))

    color = None
    if ld.get("color"):
        color = clean_text(str(ld["color"]))

    sku = None
    if ld.get("sku"):
        sku = clean_text(str(ld["sku"]))

    sizes: list[str] = []
    for s in ld.get("size") or []:
        s = clean_text(str(s))
        if s and s not in sizes:
            sizes.append(s)

    material = None
    if ld.get("material"):
        material = clean_text(str(ld["material"]))

    availability = None
    offers = ld.get("offers") or {}
    if isinstance(offers, dict):
        avail = offers.get("availability") or ""
        if "InStock" in avail:
            availability = "in_stock"
        elif "OutOfStock" in avail:
            availability = "out_of_stock"

    offers_price = None
    offers_currency = "EUR"
    if isinstance(offers, dict):
        if isinstance(offers.get("price"), (int, float)):
            offers_price = float(offers["price"])
        if offers.get("priceCurrency"):
            offers_currency = str(offers["priceCurrency"])

    price_el = soup.select_one("[data-testid='price']")
    pre_price_el = soup.select_one("[data-testid='pre-sale-price']")
    on_sale = pre_price_el is not None
    price = format_price(pre_price_el.get_text(" ", strip=True) if on_sale else (price_el.get_text(" ", strip=True) if price_el else None), None, offers_currency)
    sale = format_price(price_el.get_text(" ", strip=True) if price_el else None, None, offers_currency) if on_sale else None
    if not price and offers_price is not None:
        price = f"{offers_price:.2f}{offers_currency}"
    if price and not sale and on_sale:
        sale = price if offers_price is None else f"{offers_price:.2f}{offers_currency}"

    # ---- gallery (front / back / additional) ----
    gallery = extract_gallery_images(rsc_text)
    front_im = _gallery_pick(gallery, "from the front")
    back_im = _gallery_pick(gallery, "from the back")
    image_url = normalize_image_url(front_im["src"]) if front_im else None
    back_image_url = normalize_image_url(back_im["src"]) if back_im else None
    if not image_url and isinstance(ld.get("image"), list) and ld["image"]:
        image_url = normalize_image_url(str(ld["image"][0]))

    additional = []
    seen_assets: set[str] = set()

    def _add_additional(src: str) -> None:
        norm = normalize_image_url(src)
        asset = norm.split("?", 1)[0]
        if not norm or norm == image_url or asset in seen_assets:
            return
        seen_assets.add(asset)
        additional.append(norm)

    for im in gallery:
        _add_additional(im["src"])
    if image_url and isinstance(ld.get("image"), list):
        for u in ld["image"]:
            _add_additional(str(u))
    additional_images = " , ".join(additional) if additional else None

    product_key = None
    for u in [image_url, back_image_url] + additional:
        pk = extract_product_key(u)
        if pk:
            product_key = pk
            break

    # ---- accordions ----
    details_el = soup.select_one("[data-testid='accordion-details-description']")
    description = None
    details_list: list[str] = []
    if details_el:
        p = details_el.find("p")
        if p:
            description = clean_text(p.get_text(" ", strip=True))
        for li in details_el.select("[data-testid='product-details-list-item']"):
            t = clean_text(li.get_text(" ", strip=True))
            if t and t != product_key:
                details_list.append(t)

    material_care: list[str] = []
    mat_care_el = soup.select_one("[data-testid='accordion-materialAndCare-description']")
    if mat_care_el:
        for li in mat_care_el.select("li"):
            t = clean_text(li.get_text(" ", strip=True))
            if t and t not in material_care:
                material_care.append(t)
    care_instructions: list[str] = []
    for item in material_care:
        low = item.lower()
        if low.startswith("care"):
            care_instructions.append(item)

    size_fit: list[str] = []
    size_fit_el = soup.select_one("[data-testid='accordion-sizeAndFit-description']")
    if size_fit_el:
        for li in size_fit_el.select("li"):
            t = clean_text(li.get_text(" ", strip=True))
            if t and t not in size_fit:
                size_fit.append(t)

    # ---- gender from meta description ----
    gender = None
    meta_desc = None
    meta_el = soup.find("meta", attrs={"name": "description"})
    if meta_el and meta_el.get("content"):
        meta_desc = meta_el["content"]
        m = re.search(r"for (Men|Women|Unisex)", meta_desc, re.I)
        if m:
            g = m.group(1).lower()
            gender = "unisex" if g == "unisex" else g

    return {
        "title": title,
        "description": description,
        "color": color,
        "sku": sku,
        "product_key": product_key,
        "sizes": sizes,
        "size_fit": size_fit,
        "material": material,
        "material_care": material_care,
        "care_instructions": care_instructions,
        "details_list": details_list,
        "availability": availability,
        "price": price,
        "sale": sale,
        "on_sale": on_sale,
        "currency": offers_currency,
        "image_url": image_url,
        "back_image_url": back_image_url,
        "additional_images": additional_images,
        "gender": gender,
        "meta_description": meta_desc,
    }