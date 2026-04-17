# AI Photos

Concept site: upload a photo → get an AI artwork → order it printed and shipped across the US.

Stack: Flask · OpenAI `gpt-image-1` · Stripe Checkout (test mode) · Printify (Phase 2).

---

## Local setup (5 minutes)

```bash
cd ai-photos
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env and fill in OPENAI_API_KEY, STRIPE_SECRET_KEY, STRIPE_PUBLISHABLE_KEY
# FLASK_SECRET_KEY can be any long random string locally.

python app.py
```

Open http://localhost:5000 — upload a photo, pick a style, pick a product, pay with Stripe test card `4242 4242 4242 4242` (any future expiry, any CVC).

## Deploy to Render (10 minutes)

1. Push this folder to a GitHub repo.
2. In Render, click **New → Blueprint** and point it at your repo.
   Render will read `render.yaml` automatically.
3. When prompted, paste the three secret env vars:
   - `OPENAI_API_KEY`
   - `STRIPE_SECRET_KEY`
   - `STRIPE_PUBLISHABLE_KEY`
   (`FLASK_SECRET_KEY` is auto-generated.)
4. First build ~2 min. Then hit the live URL.

### Free tier caveats
- Instance sleeps after 15 min idle (first request ~30 s cold start).
- Files in `/static/generated/` are wiped on each deploy. For Phase 2,
  move uploads to S3 or Cloudinary.

## What's in this MVP

| Route | What it does |
|---|---|
| `/` | Landing: upload form + style picker |
| `POST /generate` | Downsizes the photo to 1024px, calls OpenAI `images.edit` with a style prompt, saves output to `static/generated/` |
| `/result` | Before/after, plus 4 product cards (canvas, framed, mug, tee) |
| `/select/<key>` | Puts the product in the session, goes to checkout |
| `/checkout` | Creates a Stripe Checkout Session (hosted) with US shipping |
| `/success` | Loads the Stripe session, calls `create_printify_order_stub` (Phase 2 seam) |
| `/admin/printify?token=<ADMIN_TOKEN>` | Read-only peek at your Printify shops + products + variant IDs. Returns 404 unless the token matches. |
| `/healthz` | For Render health checks |

## Phase 2 — flip Printify live

The seam is `create_printify_order_stub(payload)` in `app.py`.

1. Products built in Printify — done. Hit
   `http://localhost:5000/admin/printify?token=<ADMIN_TOKEN>` to list your
   shops, products, and variant IDs on one page.
2. `PRINTIFY_API_KEY` and `ADMIN_TOKEN` are already wired through `.env` and
   `render.yaml`.
3. Map each `product_key` in `PRODUCTS` (app.py) to a dict holding its Printify
   `shop_id`, `printify_product_id`, and `variant_id`.
4. **Move artwork hosting off local disk first** — upload each generated
   image to Cloudinary or S3 during `/generate`, and persist the public URL
   in the session. Printify's servers must be able to fetch the image, so
   `localhost` or Render-static paths won't work reliably.
5. Replace the stub with a real
   `POST https://api.printify.com/v1/shops/{shop_id}/orders.json` call:
   `address_to` from `payload["recipient"]`, `line_items[].product_id` and
   `variant_id` from the mapping, and the artwork's public URL attached to
   the line item.
6. Post orders as drafts first (don't immediately send to production) —
   inspect in Printify's dashboard, approve manually for the first few
   orders, then flip to auto-submit once you trust the flow.
7. Move order creation off the `/success` redirect onto a Stripe webhook
   (`checkout.session.completed`) — prevents duplicates on refresh and
   handles async payment methods.

## Known trade-offs in Hour 1

- `gpt-image-1` takes 15–30 s. The submit button shows a loading state.
- Generated images live on local disk. Fine for a demo, swap for object
  storage before Phase 2.
- No persistent database. Order metadata rides on Stripe + session.
- Product mockups are CSS tricks, not real Printify renderings.
- No webhook yet — success page handles post-payment work inline.
