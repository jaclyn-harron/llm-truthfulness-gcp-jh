# GCP Setup — exact steps

Everything below runs against a single GCP project. You need the **Owner** or
(Editor + Project IAM Admin) role on it, the `gcloud` CLI, and Terraform ≥ 1.5.
Estimated cost: tuning job ~$10-40 (one-off, dataset-size dependent);
evaluation runs pennies; the Cloud Run stack scales to zero when idle.

Throughout, replace:

- `PROJECT_ID` — your project id
- `REGION` — `us-central1` (default; any region with Vertex tuning support works,
  but keep tuning, bucket, and serving in the SAME region)
- `BUCKET` — a new bucket name, e.g. `PROJECT_ID-truthfulness-tuning`

## 0. One-time project setup

```bash
gcloud auth login
gcloud config set project PROJECT_ID

# Enable required APIs
gcloud services enable \
  aiplatform.googleapis.com \
  storage.googleapis.com \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com \
  cloudbuild.googleapis.com

# Local credentials for the code (ADC). No API keys are used anywhere.
gcloud auth application-default login
gcloud auth application-default set-quota-project PROJECT_ID

# Staging bucket for tuning data (same region as tuning/serving)
gcloud storage buckets create gs://BUCKET --location=REGION \
  --uniform-bucket-level-access
```

Your user needs these roles for Part 1 (Owners already have them):
`roles/aiplatform.user` and `roles/storage.objectAdmin` on the bucket.

## 1. Part 1 — evaluate + fine-tune

```bash
git clone git@github.com:YOUR_ORG/llm-truthfulness-gcp.git && cd llm-truthfulness-gcp
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

cp .env.example .env
# edit .env:
#   GOOGLE_CLOUD_PROJECT=PROJECT_ID
#   GOOGLE_CLOUD_LOCATION=REGION
#   TRUTHFULNESS_GCS_BUCKET=BUCKET

# Place the challenge dataset at the repo root:
cp /path/to/data.csv .

# Quick smoke test of Vertex access (~10 statements, zero-shot only):
python -m truthfulness.evaluate --data data.csv --sample 10 --skip-finetune || true

# First real run. This launches a Vertex supervised tuning job on
# gemini-2.5-flash and WAITS for it (1-3 h). Progress: Console -> Vertex AI
# -> Tuning. Tuning always uses the FULL training split; --sample 200 only
# limits the evaluation to a fixed random 200-row subset of the held-out
# test set, as a cheap end-to-end check.
python -m truthfulness.evaluate --data data.csv --sample 200

# Final evaluation: drop --sample to score both predictors on the full
# held-out test set (~1,440 rows). These are the numbers to report. The
# on-disk cache (.cache/) means already-scored rows are not re-billed.
python -m truthfulness.evaluate --data data.csv
```

When tuning finishes the script prints the tuned endpoint, e.g.
`projects/1234567890/locations/us-central1/endpoints/9876543210`.
**Save it** — every later step reuses it:

```bash
echo 'TRUTHFULNESS_FINE_TUNED_MODEL=projects/…/endpoints/…' >> .env
```

(You can also find it later: Console → Vertex AI → Tuning → your job →
tuned model endpoint, or `gcloud ai endpoints list --region=REGION`.)

## 2. Part 2 — deploy the agent service to Cloud Run

```bash
cd deploy
export TF_VAR_project_id=PROJECT_ID
export TF_VAR_region=REGION
export TF_VAR_fine_tuned_model=projects/…/locations/REGION/endpoints/…   # from step 1
export TF_VAR_api_auth_token=$(openssl rand -hex 24)
echo "API token: $TF_VAR_api_auth_token"    # save this; clients need it

make deploy
```

`make deploy` does three things: `terraform apply` targeted at the APIs +
Artifact Registry repo, `gcloud builds submit` (Cloud Build builds the image
from `deploy/Dockerfile` — `data.csv` must be at the repo root), then a full
`terraform apply` that creates:

- 5 Cloud Run services (`ts-mcp`, `ts-zero-shot`, `ts-fine-tuned`,
  `ts-explainer`, `ts-orchestrator`) + optional `ts-demo` browser UI
- a runtime service account with `roles/aiplatform.user` (this is how the
  services call Vertex — no keys)
- Secret Manager secret holding the bearer token

## 3. Verify the deployment

```bash
cd terraform
URL=$(terraform output -raw orchestrator_url)

# 401 without the token:
curl -s -o /dev/null -w '%{http_code}\n' -X POST "$URL/verify"   # -> 401

# End-to-end verify (A2A fan-out -> reconcile -> explain):
curl -s -X POST "$URL/verify" \
  -H "Authorization: Bearer $TF_VAR_api_auth_token" \
  -H "Content-Type: application/json" \
  -d '{"points":[{"statement":"Says the state budget doubled in four years.",
        "speaker_name":"A politician","statement_context":"a debate"}]}' | python -m json.tool
```

Expected response: per-statement `prediction`, `agreement` (both models'
verdicts), `confidence`, and a verdict-consistent `explanation`; plus a
`trace_id` you can grep in Cloud Run logs across all five services.

The MCP server is also usable from any generic MCP client (Claude, Cursor):
streamable-http endpoint at `$(terraform output -json agent_urls | jq -r .mcp)/mcp`,
or run `python scripts/mcp_demo.py <mcp_url>`.

## 4. Costs, teardown, rotation

```bash
# Tear the whole stack down (keeps the tuned model + bucket):
cd deploy && make destroy

# Delete the tuned model endpoint too, if finished with it:
gcloud ai endpoints list --region=REGION
gcloud ai endpoints delete ENDPOINT_ID --region=REGION

# Rotate the bearer token: re-export TF_VAR_api_auth_token and `terraform apply`.
```

## Troubleshooting

- **403 / PermissionDenied on Vertex calls** — ADC not set up
  (`gcloud auth application-default login`) or missing `roles/aiplatform.user`.
- **Tuning job rejects the base model** — the tunable set changes over time;
  as of July 2026 use `gemini-2.5-flash`, `gemini-2.5-flash-lite` or
  `gemini-2.5-pro` via `TRUTHFULNESS_FT_BASE_MODEL`. Gemini 2.5 retires
  Oct 16 2026; re-tune on the then-current tunable base afterwards.
- **Tuned endpoint NotFound from the service** — `TF_VAR_fine_tuned_model`
  region must match `GOOGLE_CLOUD_LOCATION` the services use
  (`vertex_location` Terraform var defaults to the Cloud Run region).
- **Cloud Build "data.csv not found"** — the dataset must sit at the repo root
  when you run `make deploy`; `.gcloudignore` deliberately allows it through
  (it is still never committed to git).
- **429 rate limits during evaluation** — lower `TRUTHFULNESS_MAX_CONCURRENCY`
  (default 8); calls are retried with backoff and cached, so re-running
  resumes where it left off.
