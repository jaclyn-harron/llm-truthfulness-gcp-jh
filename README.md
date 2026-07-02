# Truthfulness — LLM Predictors, Explainer & Multi-Agent Service on GCP

LLM-based binary truthfulness classification of political statements
(LIAR/PolitiFact-style data), built entirely on **Google models served from
Vertex AI in your own GCP project**. No third-party model providers, no API
keys — authentication is Application Default Credentials (ADC) end to end.

This is the internal GCP port of the original challenge solution. Same
methodology, same interfaces, Google-only models:

| Component | Model |
|---|---|
| 1. Zero-shot predictor | `gemini-2.5-flash` (Vertex AI) |
| 2. Fine-tuned predictor | Vertex AI **supervised fine-tuning** of `gemini-2.5-flash` |
| 3. Explainer | `gemini-2.5-flash` (Vertex AI) |
| Part 2 service | A2A agents + MCP tool server on Cloud Run |

Zero-shot and fine-tuned share the same base model deliberately: the
side-by-side comparison then isolates the effect of tuning rather than a
base-model difference.

> **Model lifecycle note (July 2026):** supervised tuning currently supports
> `gemini-2.5-flash`, `gemini-2.5-flash-lite` and `gemini-2.5-pro`; tuning for
> Gemini 3.x is not yet available. Google has announced an October 16, 2026
> retirement for the Gemini 2.5 stable models — check what happens to tuned
> endpoints after that date and re-tune on the then-current tunable base if
> needed (`TRUTHFULNESS_FT_BASE_MODEL`).

See **[GCP_SETUP.md](GCP_SETUP.md)** for the exact, copy-pasteable setup and
deployment steps.

## Quick start (Part 1)

```bash
# 0. Prereqs: Python 3.10+, gcloud CLI, a GCP project with billing.

# 1. Install (pinned deps)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

# 2. Authenticate (no API keys anywhere)
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID

# 3. Configure
cp .env.example .env      # set GOOGLE_CLOUD_PROJECT, TRUTHFULNESS_GCS_BUCKET

# 4. Place the challenge dataset at the repo root as data.csv (never committed)

# 5. Reproducible evaluation (held-out split, fixed seed).
#    First run launches a Vertex tuning job (~1-3 h); subsequent runs reuse it.
#    --sample N evaluates on a fixed random N-row subset of the held-out test
#    set (cheap end-to-end check); it does NOT affect tuning, which always
#    uses the full training split.
python -m truthfulness.evaluate --data data.csv --sample 200

# 6. Final numbers: drop --sample to evaluate on the full held-out test set.
#    Cached rows from step 5 are not re-billed.
python -m truthfulness.evaluate --data data.csv
```

After the tuning job completes the script prints the tuned-model endpoint.
Pin it so future runs (and the deployed service) skip training:

```bash
export TRUTHFULNESS_FINE_TUNED_MODEL=projects/…/locations/us-central1/endpoints/…
```

## Target framing (six-way → binary)

The source data carries a six-way ordinal label. It is mapped to a binary
target by splitting the ordinal scale down the middle: `true`, `mostly-true`,
`half-true` → **True**; `barely-true`, `false`, `extremely-false` → **False**
(~56% True — near balanced). False (misinformation) is the positive class for
precision/recall reporting. Exact-duplicate statements are dropped before
splitting so no text leaks across the train/val/test boundary.

## Evaluation methodology

One stratified 70/15/15 train/val/test split, fixed seed (`123`), shared by
every component. Both predictors are scored on the *same* held-out test rows;
fine-tuning only ever sees the training split (with its own internal
train/val carve-out for the tuning job's validation curve). Metrics: accuracy,
balanced accuracy, macro-F1, precision/recall/F1 on the False class, ROC-AUC
and Brier score (calibration) from the predictors' probability outputs.
All LLM calls are cached on disk (`.cache/`), so re-runs are free and
deterministic.

## The three components

```python
from truthfulness import ZeroShotPredictor, FineTunedPredictor, Explainer, load_dataset, make_splits, get_config

cfg = get_config()
df = load_dataset("data.csv")
splits = make_splits(df, seed=cfg.seed, test_size=cfg.test_size, val_size=cfg.val_size)

# 1) Zero-shot (no training of any kind)
zs = ZeroShotPredictor()
out = zs.predict(splits.test, labels=splits.test["label"])

# 2) Fine-tuned — Vertex AI supervised tuning of Gemini
ft = FineTunedPredictor()
ft.fine_tune(splits.train)            # launches + waits for the tuning job
out = ft.predict(splits.test, labels=splits.test["label"])

# 3) Explainer — works with either predictor
exp = Explainer()
explained = exp.explain(ft, splits.test.head(5))
```

## Fine-tuning approach

Vertex AI **supervised fine-tuning** (LoRA-style adapter tuning) of
`gemini-2.5-flash`. Training examples use the official tuning JSONL schema —
a `systemInstruction` plus a `contents` conversation whose final `model` turn
is the gold single word `true`/`false`. Data is staged in a GCS bucket in your
project; the tuned model is served automatically on a Vertex endpoint when the
job completes.

Training the model to emit a single word matters for calibration: at inference
we request token logprobs and softmax the `true`/`false` token probabilities,
giving a genuine probability for ROC-AUC/Brier rather than a self-reported
confidence. Thinking is disabled (`TRUTHFULNESS_THINKING_BUDGET=0`) so the
first generated token *is* the answer token.

## Explainer faithfulness

The explanation is generated *conditioned on the underlying model's actual
verdict and confidence* — it never argues for the opposite class, and it must
name concrete, statement-specific factors. A consistency guard flags any
explanation whose stated verdict drifts from the model's verdict. The
orchestrator (Part 2) uses the same mechanism to explain the *final reconciled*
verdict.

## Part 2: multi-agent service (A2A + MCP) on Cloud Run

One container image, five Cloud Run services selected by `ROLE`:

- `ts-mcp` — MCP tool server (streamable-http) exposing `predict_zero_shot`,
  `predict_fine_tuned`, `explain`, `compute_metrics`, `dataset_sample`.
- `ts-zero-shot`, `ts-fine-tuned`, `ts-explainer` — A2A agents (agent-card
  discovery + `message/send`), each consuming the MCP tools as a client.
- `ts-orchestrator` — public entry point. `POST /verify` (bearer-protected)
  fans out to both predictor agents over A2A in parallel, reconciles
  disagreements (default: defer to the fine-tuned model), asks the explainer
  agent to justify the consensus, and aggregates metrics via the MCP metric
  tool. A `trace_id` follows the request across every hop.

```bash
curl -X POST "$ORCHESTRATOR_URL/verify" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"points":[{"statement":"The unemployment rate doubled last year.","speaker_name":"..."}],"labels":null}'
```

Local end-to-end run: `cd deploy && docker compose up --build` (uses your local
ADC; see docker-compose.yml). Deployment is declarative Terraform under
`deploy/terraform/` — Cloud Run services, Secret Manager (bearer token only),
Artifact Registry, and the `roles/aiplatform.user` grant for the runtime
service account. See [GCP_SETUP.md](GCP_SETUP.md).

## Repository hygiene

`data.csv`, `.env`, caches, artifacts, tuning JSONL files and Terraform state
are gitignored and must never be committed. There are no secrets in the repo at
all: the only secret anywhere is the service's bearer token, which lives in
GCP Secret Manager.

### Layout

```
src/truthfulness/          Part 1 package (predictors, explainer, eval)
src/truthfulness_service/  Part 2 (A2A agents, MCP server, orchestrator)
deploy/                    Dockerfile, docker-compose, Makefile, terraform/
tests/                     offline tests (no network)
scripts/mcp_demo.py        generic MCP client demo
GCP_SETUP.md               exact GCP setup & deployment instructions
```

## Tests

```bash
PYTHONPATH=src pytest -q     # 14 offline tests, no GCP access needed
```
