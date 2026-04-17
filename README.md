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

### Local Stripe webhook

The Printify order is created by a Stripe webhook (`checkout.session.completed`),
not inline on `/success`. To exercise the full flow locally, run the Stripe CLI
in a second terminal:

```bash
# one-time
brew install stripe/stripe-cli/stripe   # macOS; stripe.com/docs/stripe-cli for others
stripe login

# every dev session
stripe listen --forward-to localhost:5000/webhooks/stripe
```

The CLI prints a `whsec_...` signing secret on first run — paste it into
`STRIPE_WEBHOOK_SECRET` in `.env`, then restart `python app.py`. Every completed
Stripe Checkout will now fire the webhook and create a Printify order. Skip this
step and payments will still succeed, but Printify won't see them.

## Deploy to Render (10 minutes)

1. Push this folder to a GitHub repo.
2. In Render, click **New → Blueprint** and point it at your repo.
   Render will read `render.yaml` automatically.
3. When prompted, paste the secret env vars:
   - `OPENAI_API_KEY`
   - `STRIPE_SECRET_KEY`
   - `STRIPE_PUBLISHABLE_KEY`
   - `PRINTIFY_API_KEY`
   - `STRIPE_WEBHOOK_SECRET` — *temporarily leave blank, set in step 5*
   (`FLASK_SECRET_KEY` and `ADMIN_TOKEN` are auto-generated.)
4. First build ~2 min. Hit the live URL to confirm the app boots.
5. In Stripe Dashboard → Developers → Webhooks → **Add endpoint**:
   - URL: `https://your-render-url/webhooks/stripe`
   - Events: `checkout.session.completed`
   - Copy the signing secret (`whsec_...`), paste it into `STRIPE_WEBHOOK_SECRET`
     in Render's env vars, and redeploy.

### Free tier caveats
- Instance sleeps after 15 min idle (first request ~30 s cold start).
- Files in `/static/generated/` are wiped on each deploy. For Phase 2,
  move uploads to S3 or Cloudinary.

## What's in this MVP

| Route | What it does |
|---|---|
| `/` | Landing: upload form + style picker |
| `POST /generate` | Downsizes photo, removes background, calls OpenAI `gpt-image-1` with a stability-framework prompt, saves output to `static/generated/` |
| `/result` | Before/after, plus product cards with variant picker |
| `POST /select/<key>` | Adds product + variant to the cart |
| `/cart` | Cart view — qty controls, remove, subtotal, checkout button |
| `/cart/update/<i>`, `/cart/remove/<i>`, `/cart/clear` | Cart mutations |
| `/checkout` | Creates a Stripe Checkout Session from the cart and stores a pending order record server-side |
| `/success` | Displays the order; reads fulfilment status from the order register. Does NOT create Printify orders. |
| `POST /webhooks/stripe` | Receives `checkout.session.completed`, verifies signature, creates one Printify order per completed checkout. Idempotent. |
| `/admin/printify?token=<ADMIN_TOKEN>` | Read-only peek at Printify shops + products + variant IDs. 404 without a matching token. |
| `/healthz` | Render health check |

## Known trade-offs

- `gpt-image-1` takes 15–30 s. The submit button shows a loading state.
- Generated images live on local disk. Fine for a demo, swap for S3/Cloudinary
  once you outgrow free-tier Render (files wipe on each redeploy).
- `orders.json` is also local disk. Single-instance, free-tier-friendly. Move to
  SQLite/Postgres if we scale out or need an admin order history view.
- Product mockups are CSS tricks, not Printify-rendered product photos.
- Shipping is charged at $0. Printify invoices us for real shipping — wire up
  Printify's shipping quote API before anything real launches.
