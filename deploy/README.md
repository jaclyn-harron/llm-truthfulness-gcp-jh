# deploy/

One container image (Dockerfile), every service selected by `ROLE`:
`mcp | zero_shot | fine_tuned | explainer | orchestrator | demo`.

- **Local:** `docker compose up --build` (uses your local ADC; set
  `GOOGLE_CLOUD_PROJECT` and `TRUTHFULNESS_FINE_TUNED_MODEL`).
- **GCP:** `make deploy` — Terraform under `terraform/` creates the five
  Cloud Run services, the runtime service account (granted
  `roles/aiplatform.user` for Vertex access — no API keys), and the
  Secret Manager secret holding the public endpoint's bearer token.

Exact step-by-step instructions: see [../GCP_SETUP.md](../GCP_SETUP.md).

Security note: internal services (MCP + agents) are reachable by URL but carry
no secrets; the public orchestrator requires `Authorization: Bearer <token>`.
Hardening option: switch the internal services to `ingress = INTERNAL` +
IAM-authenticated invocation and attach Google ID tokens in
`truthfulness_service/a2a_base.py` / `mcp_client.py`.
