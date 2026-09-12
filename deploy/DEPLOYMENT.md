# Deployment

Two things are worth understanding before you touch anything: **what gets
deployed**, and **why the backend kept going to sleep**.

---

## The sleep problem, and what fixes it

A free Hugging Face CPU Space sleeps after **48 hours without traffic**. It
wakes on the next HTTP request it receives, but the wake is not instant — the
container has to start, TensorFlow has to import, and the MobileNetV2 weights
have to load. That is 60–90 seconds during which the Space accepts your
connection and simply holds it.

The original setup made this worse than it had to be, in a specific way: the
dashboard and the API were **two separate Spaces**. Opening the dashboard woke
the *dashboard's* Space. It did nothing at all for the API's Space, which had
its own independent idle timer. So the page loaded, looked healthy, and then
every panel on it failed — and the only way back was to open the API Space by
hand and restart it.

Three changes address this, and they stack:

### 1. One container instead of two (the structural fix)

The root `Dockerfile` builds the React dashboard and has FastAPI serve it
alongside its own routes. One Space, one URL, one idle timer. If the page
loads, the API behind it is by definition already running — there is no second
service that can be asleep.

This is the recommended deployment and the one the README documents.

### 2. Scheduled pings (the operational fix)

`.github/workflows/keep-spaces-awake.yml` calls `/health` on each Space every
20 minutes. The 48-hour idle timer never runs out, so the Space never sleeps
and nobody ever pays a cold start.

This needs no secrets and works as soon as the file is on the default branch.

**Caveat:** GitHub disables scheduled workflows in a repository with no commits
for 60 days (it emails you first). Any push re-enables them.

### 3. Automatic restart (the recovery fix)

A ping wakes a *sleeping* Space. It cannot do anything for one that is
**paused**, has failed to build, or crashed on startup. For those the workflow
falls back to the Hub's restart endpoint:

```
POST https://huggingface.co/api/spaces/{owner}/{space}/restart
Authorization: Bearer <token>
```

To enable it, add a repository secret named `HF_TOKEN` holding a Hugging Face
access token with write access to the Spaces:

> GitHub repo → Settings → Secrets and variables → Actions → New repository
> secret → Name `HF_TOKEN`

Without the secret the restart step logs a note and skips; it does not fail the
run.

### 4. What the dashboard does while it waits

Even with all of the above, a first visitor can still arrive mid-wake — right
after a rebuild, for instance. So the dashboard opens on a wake-up screen
rather than a wall of broken panels: it explains what is happening, shows
elapsed time, retries every four seconds (which is itself what wakes a sleeping
Space), and offers a direct link to the Space for a manual restart if it takes
more than about three minutes. Once the API answers, the screen disappears and
never returns for that session — a later blip shows a thin banner instead,
because by then there is work on screen worth keeping.

---

## Deploying the single container

```bash
bash deploy/push-to-space.sh
```

That is the whole deploy. The script fetches the Space, makes its tree match
your current branch, and pushes — as a child of the Space's own HEAD, so it
fast-forwards and never needs `--force`.

The one thing it does that a plain `git push space main` cannot: a Space
identifies itself through YAML frontmatter at the top of its `README.md` (the
Docker SDK declaration and the port), and the GitHub README deliberately has
none, because GitHub renders frontmatter as a stray table above the title. The
script concatenates `deploy/space-frontmatter.md` with the real README at
deploy time, so there is only ever one README to maintain.

To deploy somewhere else, pass the Space id:

```bash
bash deploy/push-to-space.sh CholatemGiet/fashion-mnist-frontend
```

The Space builds the root `Dockerfile`, which:

- compiles the dashboard with Vite in a Node stage,
- installs the Python dependencies,
- copies the model, the dataset, the figures and the built dashboard,
- runs as uid 1000, which is what Spaces require — anything the app writes at
  runtime (the SQLite database, `models/training_metrics.json` after a
  retraining run) lives in a directory owned by that user.

Both Spaces can run this same image. Doing that makes the frontend Space
self-sufficient and removes the cross-Space dependency for good.

> **Large files:** the model and the dataset CSVs are tracked with Git LFS (see
> `.gitattributes`). Hugging Face supports LFS, but make sure `git lfs install`
> has been run locally or the Space will receive pointer files and the model
> load will fail at startup.

---

## Deploying the dashboard separately (not recommended)

If the dashboard has to be its own Space, use `Dockerfile.frontend` — Nginx
serving a static build, no Python, no model — and tell it where the API lives
at **build time**:

```
VITE_API_BASE=https://cholatemgiet-fashion-mnist-api.hf.space
```

Set it as a build-time variable in the Space settings, and point
`deploy/space-frontmatter.md`'s title at that Space before deploying to it.

Understand what you are choosing: two Spaces, two idle timers, and a cold start
that the dashboard can only wait out. The keep-awake workflow already covers
both Spaces, so this is survivable — it is just strictly worse than one
container.

---

## Local Docker stack

```bash
docker compose up --build              # dashboard + API on http://localhost
docker compose up --scale backend=3    # three API replicas behind Nginx
```

Nginx serves the built dashboard and proxies the API paths to the replica pool,
so the browser sees a single origin — the same shape as production. Locust is
pointed at Nginx rather than at one backend, so a load test exercises the load
balancer and every replica.

---

## Verifying a deployment

```bash
BASE=https://cholatemgiet-fashion-mnist-api.hf.space

curl -s $BASE/health          # {"status":"ok","model_ready":true,...}
curl -s $BASE/info            # service name, version, uptime
curl -sI $BASE/               # 200 text/html — the dashboard
curl -sI $BASE/evaluation     # 200 text/html — a deep link, not a 404
curl -s $BASE/metrics | head  # JSON, not HTML
```

The last two matter together. The dashboard's routes (`/classify`, `/training`,
`/dataset`, `/evaluation`, `/status`, `/about`) were deliberately named to avoid
colliding with API routes: both live at the root of the same origin, so a page
called `/metrics` would have been shadowed by the `/metrics` endpoint and would
have returned JSON on every refresh or shared link.
