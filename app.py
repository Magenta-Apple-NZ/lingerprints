"""
AI Photos — concept site MVP.

Flow:
  1. Visitor uploads a photo + picks a style.
  2. We call OpenAI gpt-image-1 to return an 'AI artwork'.
  3. They pick a product (canvas, framed, mug, tee).
  4. Stripe Checkout (test mode) takes payment + US shipping.
  5. Success page creates a (stubbed) Printify order — Phase 2 swaps in real API.
"""
from __future__ import annotations

import base64
import gc
import io
import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests
import stripe
from dotenv import load_dotenv
from flask import (
    Flask,
    abort,
    flash,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont, ImageOps

load_dotenv()

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-only-change-me")
# Cap uploads at 8MB — we'll downsize to 1024px below before sending to OpenAI.
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

GENERATED_DIR = Path(app.static_folder) / "generated"
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Prompt framework
#
# Every generation is assembled from three layers:
#
#   [FIDELITY_PREAMBLE]  Hard constraints applied to every style. Tells the
#                        model to preserve the subject, composition, pose,
#                        and framing of the input. This is the single most
#                        important piece for stable output.
#   [STYLE direction]    The creative / aesthetic direction (below).
#   [NEGATIVE_CONSTRAINTS] What the model must NOT do — adding extra people,
#                        text, watermarks, hallucinated limbs, etc.
#
# Combined via build_prompt() below.
# --------------------------------------------------------------------------

NEGATIVE_CONSTRAINTS = (
    "Do not include any text, letters, numbers, watermarks, logos, signatures, "
    "captions, or speech bubbles."
)

STYLES = {
    "oil": {
        "label": "Oil Painting",
        "example_breed": "German Shepherd",
        "example_breed_cat": "Maine Coon",
        "preamble": (
            "Study the breed, markings, and character of the animal in this photograph. "
            "Create a formal regal portrait in the tradition of 17th-century Dutch oil painting. "
            "Dress the subject in aristocratic human clothing — a velvet coat, lace collar, or military uniform — "
            "and seat them upright in a grand interior: dark wood panelling, draped curtains, candlelight. "
            "The subject should look noble and self-aware, as if sitting for their official portrait. "
            "Rich impasto brushwork, warm chiaroscuro lighting, fine canvas texture."
        ),
    },
    "watercolour": {
        "label": "Watercolour",
        "example_breed": "Golden Retriever",
        "example_breed_cat": "Ragdoll",
        "preamble": (
            "Identify the key features and mood of the subject in this photograph. "
            "Reimagine them as a loose, luminous watercolour illustration — "
            "soft colour bleeds, wet-on-wet edges, visible cold-press paper texture. "
            "Simplify the background to a single atmospheric wash. "
            "The style should feel like an editorial illustration from a high-end nature magazine: "
            "gestural, impressionistic, beautiful."
        ),
    },
    "van_gogh": {
        "label": "Van Gogh",
        "example_breed": "French Bulldog",
        "example_breed_cat": "Bengal",
        "preamble": (
            "Use the subject of this photograph as inspiration. "
            "Reimagine them as a Van Gogh post-impressionist painting: "
            "thick swirling impasto brushwork, vivid complementary colours, "
            "a dynamic turbulent sky or bold patterned background behind the subject. "
            "Channel the emotional intensity of 'Starry Night' and 'Portrait of Doctor Gachet'. "
            "The subject should feel alive with expressive energy."
        ),
    },
    "ghibli": {
        "label": "Ghibli Anime",
        "example_breed": "Labrador",
        "example_breed_cat": "Scottish Fold",
        "preamble": (
            "Use the subject of this photograph as inspiration. "
            "Reimagine them as a character in a Studio Ghibli film — "
            "soft painterly backgrounds with lush greens and golden light, "
            "clean expressive line work, large luminous eyes, gentle whimsical personality. "
            "Place the character in an evocative Ghibli landscape: a hillside, a forest path, or a cosy interior. "
            "The tone should feel warm, cinematic, and full of quiet wonder."
        ),
    },
    "line": {
        "label": "Line Art",
        "example_breed": "Pitbull",
        "example_breed_cat": "Siamese",
        "preamble": (
            "Study the face and most distinctive features of the subject in this photograph. "
            "Create a front-facing minimal continuous line drawing — "
            "a single unbroken black line on cream white paper that captures only the essential contours: "
            "the silhouette, the eyes, the defining characteristic features. "
            "Leave large areas of white space. Suggest rather than describe. "
            "The result should feel like a gallery-quality art print — "
            "elegant, reductive, immediately recognisable despite what is left out."
        ),
    },
}


def build_prompt(style_key: str) -> str:
    style = STYLES[style_key]
    return f"{style['preamble']}\n\n{NEGATIVE_CONSTRAINTS}"


def pick_output_size(img: Image.Image) -> str:
    """
    Match output aspect ratio to input so composition stays stable.
    gpt-image-1 supports 1024x1024, 1536x1024, 1024x1536.
    """
    w, h = img.size
    aspect = w / h if h else 1.0
    if aspect >= 1.2:
        return "1536x1024"   # landscape
    if aspect <= 0.83:
        return "1024x1536"   # portrait
    return "1024x1024"       # square-ish

PRINTIFY_SHOP_ID = "27218748"

# --------------------------------------------------------------------------
# Product mockups
#
# `mockup.url` is a publicly fetchable Printify product-photo URL. Grab them
# via the /admin/mockups?token=... diagnostic page once, paste here. Leave
# blank to fall back to the CSS-only mockups.
#
# `mockup.print_area` defines where the user's artwork is composited on the
# product photo, as percentages of the container. Tune these once per
# mockup URL you pick. Starting values are reasonable for Printify's
# default mockups but will need eyeball calibration.
#
# `mockup.blend` lets us apply `mix-blend-mode` so the artwork sits
# convincingly on fabric / canvas texture.
# --------------------------------------------------------------------------

PRODUCTS = {
    "digital_download": {
        "name": "Digital Download",
        "blurb": "Your furbaby's portrait as a PNG, delivered instantly. Share it, print it yourself, or keep it forever.",
        "digital": True,
        "blueprint_id": None,
        "print_provider_id": None,
        "printify_product_id": None,
        "print_position": None,
        "option_label": "Resolution",
        "variants": [
            {"id": 1, "label": "Standard (share & small prints)", "price_cents": 99},
            {"id": 2, "label": "Print-Ready (large canvas & frames)", "price_cents": 999},
        ],
        "mockup": {"url": "", "print_area": {}, "blend": "normal"},
    },
    "canvas": {
        "name": 'Canvas Print',
        "blurb": 'Matte gallery-wrapped canvas, frameless and ready to hang.',
        # TODO: blueprint_id + print_provider_id — refresh /admin/printify
        # (now exposes them on each product) and paste here.
        "blueprint_id": None,
        "print_provider_id": None,
        "printify_product_id": "69f261e45f78c14a7b0a4b93",
        "print_position": "front",
        "option_label": None,
        "variants": [
            {"id": 101412, "label": '10″ × 8″', "price_cents": 3900},
        ],
        "mockup": {
            "url": "",
            "print_area": {"top": 18, "left": 27, "width": 46, "height": 60},
            "blend": "multiply",
        },
    },
    "mug": {
        "name": 'Ceramic Mug',
        "blurb": 'Start every morning with their face. Dishwasher-safe ceramic.',
        # TODO: blueprint_id + print_provider_id — refresh /admin/printify
        # (now exposes them on each product) and paste here.
        "blueprint_id": None,
        "print_provider_id": None,
        "printify_product_id": "69f26147e64c9f31b70f53db",
        "print_position": "front",
        "option_label": None,
        "variants": [
            {"id": 65216, "label": "11oz", "price_cents": 2400},
        ],
        "mockup": {
            "url": "",
            "print_area": {"top": 25, "left": 30, "width": 40, "height": 50},
            "blend": "multiply",
        },
    },
    # TODO: notebook — no Printify product exists yet. Create one in the
    # API store, then add an entry here using the same shape as canvas/mug.
}

# --------------------------------------------------------------------------
# Lazy clients — so the app boots even if env vars aren't set yet.
# --------------------------------------------------------------------------


def _openai_client() -> OpenAI:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=key)


def _ensure_stripe() -> None:
    key = os.environ.get("STRIPE_SECRET_KEY")
    if not key:
        raise RuntimeError("STRIPE_SECRET_KEY is not set.")
    stripe.api_key = key


def _printify_headers() -> dict:
    key = os.environ.get("PRINTIFY_API_KEY")
    if not key:
        raise RuntimeError("PRINTIFY_API_KEY is not set.")
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def upload_artwork_to_printify(artwork_path: Path) -> str:
    """Upload artwork PNG to Printify and return the CDN preview URL."""
    contents = base64.b64encode(artwork_path.read_bytes()).decode()
    resp = requests.post(
        "https://api.printify.com/v1/uploads/images.json",
        headers=_printify_headers(),
        json={"file_name": artwork_path.name, "contents": contents},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["preview_url"]


def create_printify_order(payload: dict) -> str:
    """Submit an order to Printify and return the Printify order ID."""
    resp = requests.post(
        f"https://api.printify.com/v1/shops/{PRINTIFY_SHOP_ID}/orders.json",
        headers=_printify_headers(),
        json=payload,
        timeout=30,
    )
    if not resp.ok:
        app.logger.error("Printify order failed %s: %s", resp.status_code, resp.text)
        resp.raise_for_status()
    return resp.json()["id"]


# --------------------------------------------------------------------------
# Cart
#
# Lives in the signed Flask session cookie — no database. Each item is
# {product_key, variant_id, artwork, style, qty}. Coalesces by the combo
# of product_key + variant_id + artwork. Capped to keep the cookie under 4KB.
# --------------------------------------------------------------------------

MAX_CART_ITEMS = 10


def _find_variant(product: dict, variant_id: int) -> dict | None:
    for v in product.get("variants", []):
        if v["id"] == variant_id:
            return v
    return None


def cart_get() -> list[dict]:
    return list(session.get("cart", []))


def cart_save(cart: list[dict]) -> None:
    session["cart"] = cart
    session.modified = True


def cart_add(product_key: str, variant_id: int, artwork: str, style: str) -> tuple[bool, str]:
    """Add to cart, coalescing qty if the same line already exists."""
    if product_key not in PRODUCTS:
        return False, "Unknown product."
    product = PRODUCTS[product_key]
    variant = _find_variant(product, variant_id)
    if variant is None:
        return False, "That product option is no longer available."

    cart = cart_get()
    for item in cart:
        if (
            item["product_key"] == product_key
            and item["variant_id"] == variant_id
            and item["artwork"] == artwork
        ):
            item["qty"] = min(item.get("qty", 1) + 1, 99)
            cart_save(cart)
            return True, f"Updated quantity for {product['name']}."

    if len(cart) >= MAX_CART_ITEMS:
        return False, f"Cart limit reached ({MAX_CART_ITEMS} lines). Remove one to add more."

    cart.append({
        "product_key": product_key,
        "variant_id": variant_id,
        "artwork": artwork,
        "style": style,
        "qty": 1,
    })
    cart_save(cart)
    label = f" ({variant['label']})" if product.get("option_label") else ""
    return True, f"Added {product['name']}{label} to your cart."


def cart_remove(index: int) -> None:
    cart = cart_get()
    if 0 <= index < len(cart):
        cart.pop(index)
        cart_save(cart)


def cart_update_qty(index: int, qty: int) -> None:
    cart = cart_get()
    if 0 <= index < len(cart):
        cart[index]["qty"] = max(1, min(qty, 99))
        cart_save(cart)


def cart_clear() -> None:
    session.pop("cart", None)
    session.modified = True


def cart_items_decorated(cart: list[dict]) -> list[dict]:
    """Decorate cart items with product + variant + pricing for templates."""
    out = []
    for idx, item in enumerate(cart):
        product = PRODUCTS.get(item["product_key"], {})
        variant = _find_variant(product, item["variant_id"]) or {}
        style = STYLES.get(item["style"], {"label": "Custom"})
        unit_cents = variant.get("price_cents", 0)
        qty = item.get("qty", 1)
        out.append({
            "index": idx,
            **item,
            "product_name": product.get("name", "Product"),
            "product_blurb": product.get("blurb", ""),
            "option_label": product.get("option_label"),
            "variant_label": variant.get("label", ""),
            "style_label": style["label"],
            "unit_cents": unit_cents,
            "line_cents": unit_cents * qty,
            "is_digital": product.get("digital", False),
        })
    return out


def cart_total_cents(cart: list[dict]) -> int:
    total = 0
    for item in cart:
        product = PRODUCTS.get(item["product_key"], {})
        variant = _find_variant(product, item["variant_id"])
        if variant:
            total += variant["price_cents"] * item.get("qty", 1)
    return total


def cart_total_qty(cart: list[dict]) -> int:
    return sum(item.get("qty", 1) for item in cart)


@app.context_processor
def inject_cart_count():
    return {"cart_count": cart_total_qty(cart_get())}


# --------------------------------------------------------------------------
# Order register
#
# Server-side record of orders keyed by Stripe checkout session id. Used to:
#   1. Hand the cart snapshot to the Stripe webhook (which runs in a
#      different request context from the user's browser, so the Flask
#      session cookie is invisible to it).
#   2. Dedupe Printify order creation when Stripe retries the webhook.
#   3. Let /success display the fulfilment status once the webhook has run.
#
# Implementation: a single JSON file under static/, guarded by a process
# lock. Fine for free-tier Render (single instance). For multi-instance or
# persistent-disk setups we'd switch to SQLite or Postgres.
# --------------------------------------------------------------------------

ORDERS_FILE = Path(app.static_folder) / "orders.json"
_orders_lock = threading.Lock()


def _load_orders() -> dict:
    if not ORDERS_FILE.exists():
        return {}
    try:
        return json.loads(ORDERS_FILE.read_text())
    except Exception:
        app.logger.exception("orders.json unreadable, starting fresh")
        return {}


def _save_orders(orders: dict) -> None:
    tmp = ORDERS_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(orders, indent=2, sort_keys=True))
    tmp.replace(ORDERS_FILE)


def register_pending_order(session_id: str, cart: list[dict]) -> None:
    with _orders_lock:
        orders = _load_orders()
        orders[session_id] = {
            "status": "pending",
            "cart": cart,
            "printify_order_id": None,
            "error": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        _save_orders(orders)


def mark_order_submitted(session_id: str, printify_order_id: str) -> None:
    with _orders_lock:
        orders = _load_orders()
        if session_id in orders:
            orders[session_id]["status"] = "submitted"
            orders[session_id]["printify_order_id"] = printify_order_id
            orders[session_id]["submitted_at"] = datetime.now(timezone.utc).isoformat()
            _save_orders(orders)


def mark_order_failed(session_id: str, error: str) -> None:
    with _orders_lock:
        orders = _load_orders()
        if session_id in orders:
            orders[session_id]["status"] = "failed"
            orders[session_id]["error"] = error[:500]
            _save_orders(orders)


def get_order(session_id: str) -> dict | None:
    with _orders_lock:
        return _load_orders().get(session_id)


def is_order_processed(session_id: str) -> bool:
    order = get_order(session_id)
    return order is not None and order.get("status") in ("submitted", "failed")


# --------------------------------------------------------------------------
# Fulfilment pipeline
#
# Extracted so /webhooks/stripe is the primary caller, and /success never
# creates orders inline (the old behaviour that risked double-submitting on
# a page refresh).
# --------------------------------------------------------------------------


def fulfil_checkout(checkout_session) -> tuple[bool, str]:
    """
    Given a fully-populated Stripe checkout session, upload artworks to
    Printify and create one Printify order. Idempotent: safe to call more
    than once for the same Stripe session — second call is a no-op.

    Returns (ok, message_or_order_id).
    """
    session_id = checkout_session.id

    if is_order_processed(session_id):
        existing = get_order(session_id) or {}
        return True, existing.get("printify_order_id") or "already processed"

    record = get_order(session_id)
    if not record:
        return False, "no pending order record for this session"

    cart = record.get("cart") or []
    customer_details = checkout_session.customer_details
    shipping_address = customer_details.address if customer_details else None
    if not cart or not customer_details or not shipping_address:
        mark_order_failed(session_id, "missing cart or customer details")
        return False, "missing cart or customer details"

    items = cart_items_decorated(cart)

    physical_items = [it for it in items if not it["is_digital"]]
    digital_items = [it for it in items if it["is_digital"]]

    # All-digital order — nothing to send to Printify.
    if not physical_items:
        mark_order_submitted(session_id, "digital-only")
        app.logger.info(
            "Digital-only order: sid=%s lines=%s", session_id, len(digital_items)
        )
        return True, "digital-only"

    # Dedupe artwork uploads for physical items only.
    artwork_urls: dict[str, str] = {}
    for item in physical_items:
        artwork = item["artwork"]
        if artwork in artwork_urls:
            continue
        try:
            artwork_urls[artwork] = upload_artwork_to_printify(GENERATED_DIR / artwork)
        except Exception as exc:
            app.logger.exception("Printify image upload failed for %s", artwork)
            artwork_urls[artwork] = ""

    printify_line_items = []
    for item in physical_items:
        image_url = artwork_urls.get(item["artwork"])
        if not image_url:
            continue
        product = PRODUCTS.get(item["product_key"], {})
        if not product:
            continue
        printify_line_items.append({
            "blueprint_id": product["blueprint_id"],
            "print_provider_id": product["print_provider_id"],
            "variant_id": item["variant_id"],
            "quantity": item["qty"],
            "print_areas": {product["print_position"]: image_url},
        })

    if not printify_line_items:
        mark_order_failed(session_id, "no printable line items")
        return False, "no printable line items"

    name_parts = (customer_details.name or "").split(" ", 1)
    payload = {
        "external_id": session_id,  # Printify will reject duplicate external_ids
        "line_items": printify_line_items,
        "shipping_method": 1,
        "send_shipping_notification": False,
        "recipient": {
            "first_name": name_parts[0],
            "last_name": name_parts[1] if len(name_parts) > 1 else "",
            "email": customer_details.email or "",
            "country": shipping_address.country or "US",
            "region": shipping_address.state or "",
            "address1": shipping_address.line1 or "",
            "address2": shipping_address.line2 or "",
            "city": shipping_address.city or "",
            "zip": shipping_address.postal_code or "",
        },
    }

    try:
        printify_order_id = create_printify_order(payload)
    except Exception as exc:
        app.logger.exception("Printify order creation failed for %s", session_id)
        mark_order_failed(session_id, f"{exc.__class__.__name__}: {exc}")
        return False, str(exc)

    mark_order_submitted(session_id, printify_order_id)
    app.logger.info(
        "Printify order submitted: sid=%s printify=%s lines=%s",
        session_id, printify_order_id, len(printify_line_items),
    )
    return True, printify_order_id


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Pets helpers
# --------------------------------------------------------------------------

MAX_PETS = 10


def pets_get() -> list[dict]:
    return list(session.get("pets", []))


def pets_save(pets: list[dict]) -> None:
    session["pets"] = pets
    session.modified = True


def pet_find(pet_id: str) -> dict | None:
    return next((p for p in pets_get() if p["id"] == pet_id), None)


@app.route("/")
def index():
    original = session.get("original")
    has_previous = bool(original and (GENERATED_DIR / original).exists())
    artwork = session.get("artwork")
    has_artwork = bool(artwork and (GENERATED_DIR / artwork).exists())
    pets = pets_get()
    return render_template(
        "index.html",
        styles=STYLES,
        has_previous=has_previous,
        previous_original=original if has_previous else None,
        has_artwork=has_artwork,
        pets=pets,
    )


@app.route("/use-last")
def use_last():
    """Dev shortcut — skip OpenAI and jump straight to /result with the last artwork."""
    artwork = session.get("artwork")
    if not artwork or not (GENERATED_DIR / artwork).exists():
        flash("No previous artwork in session — paint one first.")
        return redirect(url_for("index"))
    return redirect(url_for("result"))


@app.route("/pets")
def pets_page():
    return render_template("pets.html", pets=pets_get())


@app.route("/pets/add", methods=["POST"])
def pets_add():
    name = request.form.get("name", "").strip() or "My Pet"
    file = request.files.get("photo")
    if not file or not file.filename:
        flash("Please choose a photo to upload.")
        return redirect(url_for("pets_page"))

    try:
        img = Image.open(file.stream)
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGBA")
    except Exception:
        flash("That file doesn't look like an image we can read. Try JPG or PNG.")
        return redirect(url_for("pets_page"))

    img.thumbnail((1024, 1024))
    filename = f"pet_{uuid.uuid4().hex}.png"
    img.save(GENERATED_DIR / filename, format="PNG")

    current = pets_get()
    if len(current) >= MAX_PETS:
        flash(f"You can save up to {MAX_PETS} pets. Remove one to add another.")
        return redirect(url_for("pets_page"))

    current.append({"id": uuid.uuid4().hex, "name": name, "filename": filename})
    pets_save(current)
    flash(f"{name} added to your pets!")
    return redirect(url_for("pets_page"))


@app.route("/pets/remove/<pet_id>", methods=["POST"])
def pets_remove(pet_id: str):
    current = pets_get()
    pet = next((p for p in current if p["id"] == pet_id), None)
    if pet:
        try:
            (GENERATED_DIR / pet["filename"]).unlink(missing_ok=True)
        except Exception:
            pass
        pets_save([p for p in current if p["id"] != pet_id])
    return redirect(url_for("pets_page"))


def _load_image_for_generation(pet_id: str | None, file) -> tuple[Image.Image | None, str]:
    """
    Return (img, error_message). Loads from a saved pet if pet_id given,
    otherwise from the uploaded file.
    """
    if pet_id:
        pet = pet_find(pet_id)
        if not pet:
            return None, "That pet wasn't found. Please select another."
        path = GENERATED_DIR / pet["filename"]
        if not path.exists():
            return None, "That pet's photo has expired. Please re-upload them."
        try:
            img = Image.open(path).convert("RGBA")
            return img, ""
        except Exception:
            return None, "Couldn't read that pet's photo. Please re-upload."

    if not file or not file.filename:
        return None, "Please select a pet or upload a photo first."
    try:
        img = Image.open(file.stream)
        img = ImageOps.exif_transpose(img)
        return img.convert("RGBA"), ""
    except Exception:
        return None, "That file doesn't look like an image we can read. Try JPG or PNG."


@app.route("/generate", methods=["POST"])
def generate():
    pet_id = request.form.get("pet_id", "").strip() or None
    style_key = request.form.get("style", "oil")
    if style_key not in STYLES:
        style_key = "oil"

    img, err = _load_image_for_generation(pet_id, request.files.get("photo"))
    if img is None:
        flash(err)
        return redirect(url_for("index"))

    # Downsize long side to 1024 max — aligns with gpt-image-1's smallest
    # output size and keeps memory well under Render's 512MB ceiling.
    # Aspect ratio is preserved.
    img.thumbnail((1024, 1024))

    # Save the original so the result page can show before/after.
    # If painting from a saved pet, re-use their file as the "original".
    if pet_id:
        original_filename = pet_find(pet_id)["filename"]
    else:
        original_filename = f"orig_{uuid.uuid4().hex}.png"
        img.save(GENERATED_DIR / original_filename, format="PNG")

    # Encode to PNG bytes — pass as (filename, bytes, mime) tuple, which is the
    # most reliable shape for the OpenAI SDK's multipart upload.
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    prompt = build_prompt(style_key)
    output_size = pick_output_size(img)

    app.logger.info(
        "GENERATE style=%s input_size=%sx%s output_size=%s prompt_len=%s",
        style_key, img.size[0], img.size[1], output_size, len(prompt),
    )

    try:
        client = _openai_client()
        result = client.images.edit(
            model="gpt-image-1",
            image=("input.png", png_bytes, "image/png"),
            prompt=prompt,
            size=output_size,
            n=1,
            input_fidelity="high",  # keep the input image structure intact
            quality="medium",
        )
        b64 = result.data[0].b64_json
        if not b64:
            raise RuntimeError("OpenAI returned no image data.")
        image_bytes = base64.b64decode(b64)
    except Exception as exc:
        app.logger.exception("OpenAI image generation failed")
        flash(
            "Our AI artist couldn't paint that one. "
            f"({exc.__class__.__name__}: {exc})"
        )
        return redirect(url_for("index"))

    artwork_filename = f"art_{uuid.uuid4().hex}.png"
    (GENERATED_DIR / artwork_filename).write_bytes(image_bytes)

    # Free the large transient buffers before returning. Render's 512MB
    # ceiling is tight enough that we don't want to wait for Python's GC
    # to decide it's time.
    del png_bytes, buf, img, image_bytes, result
    gc.collect()

    session["artwork"] = artwork_filename
    session["original"] = original_filename
    session["style"] = style_key
    session["pet_name"] = request.form.get("pet_name", "").strip()
    session["pet_type"] = request.form.get("pet_type", "dog")
    return redirect(url_for("result"))


@app.route("/regenerate", methods=["POST"])
def regenerate():
    """Re-run generation on the stored original with a different style."""
    style_key = request.form.get("style", "oil")
    if style_key not in STYLES:
        style_key = "oil"

    original = session.get("original")
    if not original:
        flash("No photo in session — please upload one.")
        return redirect(url_for("index"))

    original_path = GENERATED_DIR / original
    if not original_path.exists():
        flash("Your previous photo has expired. Please upload again.")
        return redirect(url_for("index"))

    try:
        img = Image.open(original_path)
    except Exception:
        flash("Couldn't re-read your photo. Please upload again.")
        return redirect(url_for("index"))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    prompt = build_prompt(style_key)
    output_size = pick_output_size(img)

    app.logger.info(
        "REGENERATE style=%s input_size=%sx%s output_size=%s prompt_len=%s",
        style_key, img.size[0], img.size[1], output_size, len(prompt),
    )

    try:
        client = _openai_client()
        result_obj = client.images.edit(
            model="gpt-image-1",
            image=("input.png", png_bytes, "image/png"),
            prompt=prompt,
            size=output_size,
            n=1,
            input_fidelity="high",
            quality="medium",
        )
        b64 = result_obj.data[0].b64_json
        if not b64:
            raise RuntimeError("OpenAI returned no image data.")
        image_bytes = base64.b64decode(b64)
    except Exception as exc:
        app.logger.exception("OpenAI regeneration failed")
        flash(
            "Our AI artist couldn't paint that one. "
            f"({exc.__class__.__name__}: {exc})"
        )
        return redirect(url_for("result"))

    artwork_filename = f"art_{uuid.uuid4().hex}.png"
    (GENERATED_DIR / artwork_filename).write_bytes(image_bytes)

    del png_bytes, buf, image_bytes, result_obj
    gc.collect()

    session["artwork"] = artwork_filename
    session["style"] = style_key
    return redirect(url_for("result"))


@app.route("/result")
def result():
    artwork = session.get("artwork")
    original = session.get("original")
    style_key = session.get("style")
    if not artwork:
        return redirect(url_for("index"))

    pet_name = session.get("pet_name", "")
    pet_type = session.get("pet_type", "dog")

    return render_template(
        "result.html",
        artwork=artwork,
        original=original,
        style=STYLES.get(style_key, {"label": "Custom"}),
        style_key=style_key,
        styles=STYLES,
        products=PRODUCTS,
        pet_name=pet_name,
        pet_type=pet_type,
    )


@app.route("/select/<product_key>", methods=["POST"])
def select_product(product_key: str):
    """Add a product + variant to the cart for the current artwork."""
    if product_key not in PRODUCTS:
        abort(404)
    artwork = session.get("artwork")
    if not artwork:
        return redirect(url_for("index"))
    product = PRODUCTS[product_key]
    variant_id = request.form.get("variant_id", type=int)
    # default to first variant if none supplied or unknown
    if variant_id is None or _find_variant(product, variant_id) is None:
        variant_id = product["variants"][0]["id"]

    ok, message = cart_add(
        product_key=product_key,
        variant_id=variant_id,
        artwork=artwork,
        style=session.get("style", ""),
    )
    flash(message)
    # If the user clicked "Buy now" we take them straight to checkout,
    # otherwise back to the result page so they can keep adding.
    if ok and request.form.get("action") == "buy_now":
        return redirect(url_for("cart_view"))
    return redirect(url_for("result"))


@app.route("/cart")
def cart_view():
    cart = cart_get()
    items = cart_items_decorated(cart)
    return render_template(
        "cart.html",
        items=items,
        total_cents=cart_total_cents(cart),
        is_empty=len(cart) == 0,
    )


@app.route("/cart/remove/<int:index>", methods=["POST"])
def cart_remove_route(index: int):
    cart_remove(index)
    flash("Item removed from cart.")
    return redirect(url_for("cart_view"))


@app.route("/cart/update/<int:index>", methods=["POST"])
def cart_update_route(index: int):
    qty = request.form.get("qty", type=int) or 1
    cart_update_qty(index, qty)
    return redirect(url_for("cart_view"))


@app.route("/cart/clear", methods=["POST"])
def cart_clear_route():
    cart_clear()
    flash("Cart cleared.")
    return redirect(url_for("index"))


@app.route("/checkout", methods=["GET", "POST"])
def checkout():
    cart = cart_get()
    if not cart:
        flash("Your cart is empty.")
        return redirect(url_for("index"))

    items = cart_items_decorated(cart)

    # Build Stripe line_items from the cart.
    line_items = []
    for item in items:
        name = f"Linger Prints — {item['product_name']}"
        if item["option_label"] and item["variant_label"]:
            name += f" ({item['variant_label']})"
        line_items.append({
            "price_data": {
                "currency": "usd",
                "product_data": {
                    "name": name,
                    "description": item["product_blurb"],
                },
                "unit_amount": item["unit_cents"],
            },
            "quantity": item["qty"],
        })

    has_physical = any(
        not PRODUCTS.get(item["product_key"], {}).get("digital") for item in cart
    )

    try:
        _ensure_stripe()
        create_kwargs: dict = dict(
            mode="payment",
            line_items=line_items,
            metadata={
                # Lightweight marker only — full cart is stashed in the Flask
                # session under 'pending' keyed by the Stripe session id so the
                # cookie survives the Stripe redirect round-trip.
                "cart_line_count": str(len(cart)),
            },
            success_url=url_for("success", _external=True) + "?session_id={CHECKOUT_SESSION_ID}",
            cancel_url=url_for("cart_view", _external=True),
        )
        if has_physical:
            create_kwargs["shipping_address_collection"] = {
                "allowed_countries": ["US", "NZ", "AU", "GB", "CA"]
            }
        checkout_session = stripe.checkout.Session.create(**create_kwargs)
    except Exception as exc:
        app.logger.exception("Stripe checkout create failed")
        flash(f"Checkout unavailable right now. ({exc.__class__.__name__})")
        return redirect(url_for("cart_view"))

    # Persist the cart server-side keyed by the Stripe session id. This is
    # the record the webhook will read when it fires — the Flask session
    # cookie is invisible to the webhook because the webhook request comes
    # from Stripe, not the customer's browser.
    register_pending_order(checkout_session.id, cart)

    # Also keep a hint in the user's session cookie so /success knows which
    # checkout just happened (for cart-clearing UX, not for fulfilment).
    session["pending_sid"] = checkout_session.id
    session.modified = True

    return redirect(checkout_session.url, code=303)


@app.route("/success")
def success():
    """
    Thank-you page. Does NOT create the Printify order — that's the
    webhook's job. We just display what the customer paid for and show
    whatever fulfilment state the webhook has reached.
    """
    session_id = request.args.get("session_id")
    if not session_id:
        return redirect(url_for("index"))

    try:
        _ensure_stripe()
        checkout_session = stripe.checkout.Session.retrieve(
            session_id, expand=["line_items", "customer_details"]
        )
    except Exception as exc:
        app.logger.exception("Stripe session retrieve failed")
        flash(f"We couldn't load your order. ({exc.__class__.__name__})")
        return redirect(url_for("index"))

    customer_details = checkout_session.customer_details
    shipping_address = customer_details.address if customer_details else None

    # Read fulfilment state from the order register. If the webhook has
    # already fired, we'll see status="submitted" and a printify order id.
    # Otherwise it's "pending" and we tell the customer we're processing.
    order_record = get_order(session_id) or {}
    cart_snapshot = order_record.get("cart") or []
    items = cart_items_decorated(cart_snapshot)
    total_cents = sum(item["line_cents"] for item in items)

    fulfilment_status = order_record.get("status", "unknown")
    printify_order_id = order_record.get("printify_order_id")
    display_order_id = (
        printify_order_id
        if printify_order_id and printify_order_id != "digital-only"
        else f"AIP-{session_id[-8:].upper()}"
    )
    has_digital = any(item.get("is_digital") for item in items)
    has_physical = any(not item.get("is_digital") for item in items)

    # Clear the user's cart now that the payment is confirmed. Guarded on
    # the session id so a user who lands on someone else's /success link
    # doesn't have their cart wiped.
    if session.get("pending_sid") == session_id:
        session.pop("pending_sid", None)
        cart_clear()

    return render_template(
        "success.html",
        order_id=display_order_id,
        items=items,
        total_cents=total_cents,
        customer_email=customer_details.email if customer_details else None,
        address=shipping_address,
        printify_ok=(fulfilment_status == "submitted"),
        fulfilment_status=fulfilment_status,
        has_digital=has_digital,
        has_physical=has_physical,
    )


@app.route("/webhooks/stripe", methods=["POST"])
def stripe_webhook():
    """
    Stripe calls this when a checkout completes. Verifies the signature,
    then hands off to fulfil_checkout() which creates the Printify order.
    Idempotent: repeated webhook fires for the same session are a no-op.
    """
    webhook_secret = os.environ.get("STRIPE_WEBHOOK_SECRET")
    if not webhook_secret:
        app.logger.error("STRIPE_WEBHOOK_SECRET is not set")
        return "webhook secret not configured", 500

    payload = request.data
    sig_header = request.headers.get("Stripe-Signature", "")

    try:
        _ensure_stripe()
        event = stripe.Webhook.construct_event(payload, sig_header, webhook_secret)
    except ValueError:
        app.logger.warning("Stripe webhook: invalid payload")
        return "invalid payload", 400
    except stripe.error.SignatureVerificationError:
        app.logger.warning("Stripe webhook: bad signature")
        return "bad signature", 400

    event_type = event["type"]
    app.logger.info("Stripe webhook event: %s", event_type)

    if event_type == "checkout.session.completed":
        session_id = event["data"]["object"]["id"]
        # Re-retrieve with expansions so we have customer_details + address.
        try:
            checkout_session = stripe.checkout.Session.retrieve(
                session_id, expand=["customer_details"]
            )
        except Exception:
            app.logger.exception("Stripe retrieve failed in webhook for %s", session_id)
            return "", 200  # ack so Stripe doesn't retry forever

        ok, detail = fulfil_checkout(checkout_session)
        if not ok:
            app.logger.warning("Fulfilment failed for %s: %s", session_id, detail)

    # Always 200 to a valid, signed event so Stripe stops retrying.
    return "", 200




# --------------------------------------------------------------------------
# Admin / diagnostic — read-only peek at Printify catalog so Phase 2 starts
# with shop ID + variant IDs already in hand.
# Gated by ADMIN_TOKEN so it's safe to expose on Render.
# --------------------------------------------------------------------------


@app.route("/admin/printify")
def admin_printify():
    admin_token = os.environ.get("ADMIN_TOKEN")
    if not admin_token or request.args.get("token") != admin_token:
        abort(404)

    api_key = os.environ.get("PRINTIFY_API_KEY")
    if not api_key:
        return "PRINTIFY_API_KEY is not set.", 500

    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        shops_resp = requests.get(
            "https://api.printify.com/v1/shops.json", headers=headers, timeout=15
        )
        shops_resp.raise_for_status()
        shops = shops_resp.json()

        results = []
        for shop in shops:
            products_resp = requests.get(
                f"https://api.printify.com/v1/shops/{shop['id']}/products.json?limit=50",
                headers=headers,
                timeout=15,
            )
            products_resp.raise_for_status()
            products = products_resp.json().get("data", [])
            results.append({"shop": shop, "products": products})
    except Exception as exc:
        app.logger.exception("Printify admin fetch failed")
        return f"Printify API error: {exc}", 502

    return render_template("admin_printify.html", results=results)


@app.route("/admin/mockups")
def admin_mockups():
    """
    Diagnostic page: for each product in PRODUCTS, fetch its Printify record
    and display every mockup image URL Printify has for it. Pick the one
    you like, paste it into PRODUCTS[<key>]['mockup']['url'] in app.py.

    Also shows a live preview of the print_area overlay using the
    currently-configured print_area values — so you can tune them by eye.
    """
    admin_token = os.environ.get("ADMIN_TOKEN")
    if not admin_token or request.args.get("token") != admin_token:
        abort(404)

    api_key = os.environ.get("PRINTIFY_API_KEY")
    if not api_key:
        return "PRINTIFY_API_KEY is not set.", 500

    headers = {"Authorization": f"Bearer {api_key}"}
    results = []
    for key, product in PRODUCTS.items():
        entry = {
            "key": key,
            "name": product["name"],
            "configured_mockup": product.get("mockup", {}).get("url") or "",
            "print_area": product.get("mockup", {}).get("print_area") or {},
            "images": [],
            "error": None,
        }
        try:
            resp = requests.get(
                f"https://api.printify.com/v1/shops/{PRINTIFY_SHOP_ID}/products/"
                f"{product['printify_product_id']}.json",
                headers=headers,
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            # Each image entry has 'src', 'position', 'variant_ids', 'is_default'.
            entry["images"] = data.get("images", [])
        except Exception as exc:
            entry["error"] = str(exc)
        results.append(entry)

    return render_template("admin_mockups.html", results=results)


def _watermarked_preview(img: Image.Image) -> io.BytesIO:
    """Return a JPEG BytesIO with a diagonal Linger Prints watermark tiled across the image."""
    img = img.convert("RGBA")
    w, h = img.size

    # Build watermark on a square canvas large enough to survive 45° rotation without clipping.
    diag = int((w ** 2 + h ** 2) ** 0.5) + 40
    tile = Image.new("RGBA", (diag, diag), (0, 0, 0, 0))
    draw = ImageDraw.Draw(tile)

    try:
        font = ImageFont.load_default(size=20)
    except TypeError:
        font = ImageFont.load_default()

    text = "LINGER PRINTS"
    step_x, step_y = 220, 90
    for row, y in enumerate(range(0, diag, step_y)):
        x_offset = (step_x // 2) if row % 2 else 0
        for x in range(-step_x + x_offset, diag + step_x, step_x):
            draw.text((x, y), text, fill=(255, 255, 255, 65), font=font)

    tile = tile.rotate(45)
    pad_x = (diag - w) // 2
    pad_y = (diag - h) // 2
    tile = tile.crop((pad_x, pad_y, pad_x + w, pad_y + h))

    out = Image.alpha_composite(img, tile).convert("RGB")
    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=88)
    buf.seek(0)
    return buf


@app.route("/preview/<artwork>")
def preview_artwork(artwork: str):
    """Watermarked JPEG used on the result page — clean file is reserved for purchase."""
    if "/" in artwork or "\\" in artwork or not artwork.endswith(".png"):
        abort(400)
    path = GENERATED_DIR / artwork
    if not path.exists():
        abort(404)
    buf = _watermarked_preview(Image.open(path))
    return send_file(buf, mimetype="image/jpeg")


@app.route("/download/<artwork>")
def download_artwork(artwork: str):
    """
    Serve a digital download. variant=print upscales 2.5× for large prints;
    default serves the generated PNG as-is.
    """
    # Prevent path traversal — artwork filenames are UUIDs with no slashes.
    if "/" in artwork or "\\" in artwork or not artwork.endswith(".png"):
        abort(400)

    path = GENERATED_DIR / artwork
    if not path.exists():
        abort(404)

    variant = request.args.get("variant", "standard")
    if variant == "print":
        img = Image.open(path)
        w, h = img.size
        img = img.resize((int(w * 2.5), int(h * 2.5)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return send_file(
            buf,
            mimetype="image/png",
            as_attachment=True,
            download_name="furbaby-portrait-print.png",
        )

    return send_file(path, as_attachment=True, download_name="furbaby-portrait.png")


@app.route("/healthz")
def healthz():
    return "ok", 200


@app.route("/faq")
def faq():
    return render_template("faq.html")


@app.route("/terms")
def terms():
    return render_template("terms.html")


@app.route("/privacy")
def privacy():
    return render_template("privacy.html")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
