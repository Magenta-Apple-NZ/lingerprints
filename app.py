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
import io
import logging
import os
import uuid
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
    session,
    url_for,
)
from openai import OpenAI
from PIL import Image, ImageOps
from rembg import remove as rembg_remove

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
        "preamble": (
            "Use the subject of this photograph as inspiration. "
            "Reimagine them as a character in a Studio Ghibli film — "
            "soft painterly backgrounds with lush greens and golden light, "
            "clean expressive line work, large luminous eyes, gentle whimsical personality. "
            "Place the character in an evocative Ghibli landscape: a hillside, a forest path, or a cosy interior. "
            "The tone should feel warm, cinematic, and full of quiet wonder."
        ),
    },
    "pop": {
        "label": "Pop Art",
        "preamble": (
            "Use the face and key features of the subject in this photograph. "
            "Create an extreme Andy Warhol-style silkscreen pop art piece: "
            "a 2x2 grid of the same portrait, each panel in a radically different flat colour palette — "
            "hot pink, electric blue, acid yellow, lime green. "
            "Bold black outlines, coarse halftone dot texture, zero shading. "
            "Pure graphic impact. Think the Marilyn Monroe series but with this subject."
        ),
    },
    "line": {
        "label": "Line Art",
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

PRINTIFY_SHOP_ID = "27218866"

PRODUCTS = {
    "framed_canvas": {
        "name": 'Framed Canvas Print',
        "blurb": '12×16" gallery-wrapped canvas in a premium frame.',
        "blueprint_id": 944,
        "print_provider_id": 99,
        "printify_product_id": "69e1f53cf03babfb2f075654",
        "print_position": "front",
        "option_label": "Frame Colour",
        "variants": [
            {"id": 88292,  "label": "Black",    "price_cents": 9900},
            {"id": 107252, "label": "Espresso",  "price_cents": 9900},
            {"id": 244029, "label": "Natural",   "price_cents": 9900},
            {"id": 107253, "label": "White",     "price_cents": 9900},
        ],
    },
    "stretched_canvas": {
        "name": 'Stretched Canvas Print',
        "blurb": '12×16" gallery-wrapped canvas, frameless and ready to hang.',
        "blueprint_id": 1159,
        "print_provider_id": 99,
        "printify_product_id": "69e1f4942ed6a28c54063562",
        "print_position": "front",
        "option_label": None,
        "variants": [
            {"id": 91643, "label": '12×16"', "price_cents": 5900},
        ],
    },
    "poster": {
        "name": 'Rolled Poster',
        "blurb": '12×16" high-quality print, rolled for safe delivery.',
        "blueprint_id": 1220,
        "print_provider_id": 99,
        "printify_product_id": "69e1f462596268421f0a48b9",
        "print_position": "front",
        "option_label": "Finish",
        "variants": [
            {"id": 101883, "label": "Matte",       "price_cents": 3500},
            {"id": 101832, "label": "Semi Glossy", "price_cents": 3500},
        ],
    },
    "tee": {
        "name": 'Unisex T-Shirt',
        "blurb": 'Soft 100% cotton. Your artwork on the front.',
        "blueprint_id": 12,
        "print_provider_id": 99,
        "printify_product_id": "69e1f427820f4311710d16ba",
        "print_position": "front",
        "option_label": "Size",
        "variants": [
            {"id": 18540, "label": "S",   "price_cents": 3900},
            {"id": 18541, "label": "M",   "price_cents": 3900},
            {"id": 18542, "label": "L",   "price_cents": 3900},
            {"id": 18543, "label": "XL",  "price_cents": 3900},
            {"id": 18544, "label": "2XL", "price_cents": 4500},
            {"id": 18545, "label": "3XL", "price_cents": 4900},
        ],
    },
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
# Routes
# --------------------------------------------------------------------------


@app.route("/")
def index():
    return render_template("index.html", styles=STYLES)


@app.route("/generate", methods=["POST"])
def generate():
    file = request.files.get("photo")
    style_key = request.form.get("style", "oil")

    if not file or not file.filename:
        flash("Please choose a photo first.")
        return redirect(url_for("index"))

    if style_key not in STYLES:
        style_key = "oil"

    try:
        img = Image.open(file.stream)
        # Honour EXIF orientation before any other processing.
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGBA")
    except Exception:
        flash("That file doesn't look like an image we can read. Try JPG or PNG.")
        return redirect(url_for("index"))

    # Downsize long side to 1536 max — aligns with gpt-image-1's output sizes
    # and keeps uploads small. Aspect ratio is preserved.
    img.thumbnail((1536, 1536))

    # Remove background and composite subject onto a clean white canvas.
    # This helps gpt-image-1 focus on the subject rather than cluttered
    # backgrounds, and gives the style transfer a cleaner base to work from.
    try:
        fg = rembg_remove(img)          # RGBA with transparent background
        white = Image.new("RGBA", fg.size, (255, 255, 255, 255))
        white.paste(fg, mask=fg.split()[3])
        img = white
    except Exception:
        app.logger.warning("Background removal failed — continuing with original.")
        # Non-fatal: just use the original image.

    # Save the original so the result page can show before/after.
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

    session["artwork"] = artwork_filename
    session["original"] = original_filename
    session["style"] = style_key
    return redirect(url_for("result"))


@app.route("/result")
def result():
    artwork = session.get("artwork")
    original = session.get("original")
    style_key = session.get("style")
    if not artwork:
        return redirect(url_for("index"))

    return render_template(
        "result.html",
        artwork=artwork,
        original=original,
        style=STYLES.get(style_key, {"label": "Custom"}),
        products=PRODUCTS,
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
        name = f"AI Artwork — {item['product_name']}"
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

    try:
        _ensure_stripe()
        checkout_session = stripe.checkout.Session.create(
            mode="payment",
            line_items=line_items,
            shipping_address_collection={"allowed_countries": ["US", "NZ", "AU", "GB", "CA"]},
            metadata={
                # Lightweight marker only — full cart is stashed in the Flask
                # session under 'pending' keyed by the Stripe session id so the
                # cookie survives the Stripe redirect round-trip.
                "cart_line_count": str(len(cart)),
            },
            success_url=url_for("success", _external=True) + "?session_id={CHECKOUT_SESSION_ID}",
            cancel_url=url_for("cart_view", _external=True),
        )
    except Exception as exc:
        app.logger.exception("Stripe checkout create failed")
        flash(f"Checkout unavailable right now. ({exc.__class__.__name__})")
        return redirect(url_for("cart_view"))

    # Stash the cart snapshot so /success can rebuild the order even though
    # we intentionally don't round-trip the full cart through Stripe metadata.
    session["pending"] = {"sid": checkout_session.id, "cart": cart}
    session.modified = True

    return redirect(checkout_session.url, code=303)


@app.route("/success")
def success():
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

    # Pull the cart snapshot that /checkout stashed before redirecting to Stripe.
    # Guard: only accept the snapshot if it matches the Stripe session we just
    # returned from (defends against stale pending state in the cookie).
    pending = session.get("pending") or {}
    cart_snapshot: list[dict] = []
    if pending.get("sid") == checkout_session.id:
        cart_snapshot = list(pending.get("cart", []))

    items = cart_items_decorated(cart_snapshot)
    order_id = f"AIP-{uuid.uuid4().hex[:8].upper()}"  # fallback

    printify_ok = False
    if items and customer_details and shipping_address:
        # Dedupe artwork uploads: if the same artwork appears in multiple
        # line items we only hit Printify's /uploads/images.json once.
        artwork_urls: dict[str, str] = {}
        for item in items:
            artwork = item["artwork"]
            if artwork in artwork_urls:
                continue
            try:
                artwork_urls[artwork] = upload_artwork_to_printify(
                    GENERATED_DIR / artwork
                )
            except Exception:
                app.logger.exception("Printify image upload failed for %s", artwork)
                artwork_urls[artwork] = ""

        # Build one Printify order with all line items.
        printify_line_items = []
        for item in items:
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

        if printify_line_items:
            name_parts = (customer_details.name or "").split(" ", 1)
            printify_payload = {
                "external_id": checkout_session.id,
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
                order_id = create_printify_order(printify_payload)
                printify_ok = True
                app.logger.info(
                    "Printify order created: %s (lines=%s)",
                    order_id, len(printify_line_items),
                )
            except Exception:
                app.logger.exception("Printify order creation failed")

    # Whether Printify succeeded or not, the customer paid — clear their cart
    # and pending snapshot so a refresh can't double-submit the order.
    if pending.get("sid") == checkout_session.id:
        session.pop("pending", None)
        cart_clear()

    total_cents = sum(item["line_cents"] for item in items)

    return render_template(
        "success.html",
        order_id=order_id,
        items=items,
        total_cents=total_cents,
        customer_email=customer_details.email if customer_details else None,
        address=shipping_address,
        printify_ok=printify_ok,
    )




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


@app.route("/healthz")
def healthz():
    return "ok", 200


if __name__ == "__main__":
    app.run(debug=True, port=5000)
