data "google_project" "this" {}

locals {
  # Vertex location for model calls; defaults to the Cloud Run region.
  vertex_location = var.vertex_location != "" ? var.vertex_location : var.region

  # The agents and orchestrator are wired to peers using each service's REAL
  # Cloud Run URL (.uri), so this works regardless of the project's URL format.
  # The MCP server is a standalone resource; the agents reference it; the
  # orchestrator references the agents — a clean dependency DAG, no cycles.
  mcp_url = "${google_cloud_run_v2_service.mcp.uri}/mcp"

  # Env vars shared by every service that calls Vertex AI. Authentication is
  # the runtime service account (ADC) — no API keys anywhere.
  vertex_env = {
    GOOGLE_CLOUD_PROJECT           = var.project_id
    GOOGLE_CLOUD_LOCATION          = local.vertex_location
    TRUTHFULNESS_FINE_TUNED_MODEL  = var.fine_tuned_model
  }
}

# ---- APIs ----
resource "google_project_service" "apis" {
  for_each = toset([
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "secretmanager.googleapis.com",
    "cloudbuild.googleapis.com",
    "aiplatform.googleapis.com",
  ])
  service            = each.value
  disable_on_destroy = false
}

# ---- Artifact Registry (holds the container image) ----
resource "google_artifact_registry_repository" "repo" {
  location      = var.region
  repository_id = "truthfulness"
  format        = "DOCKER"
  depends_on    = [google_project_service.apis]
}

# ---- Secret (bearer token for the public endpoint) ----
resource "google_secret_manager_secret" "auth" {
  secret_id = "truthfulness-api-auth-token"
  replication {
    auto {}
  }
  depends_on = [google_project_service.apis]
}
resource "google_secret_manager_secret_version" "auth" {
  secret      = google_secret_manager_secret.auth.id
  secret_data = var.api_auth_token
}

# ---- Runtime service account ----
resource "google_service_account" "runtime" {
  account_id   = "cloud-run-sa"
  display_name = "Truthfulness Cloud Run runtime"
}

# Vertex AI access for every model call (zero-shot, tuned endpoint, explainer).
resource "google_project_iam_member" "vertex_user" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.runtime.email}"
}

resource "google_secret_manager_secret_iam_member" "auth_access" {
  secret_id = google_secret_manager_secret.auth.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.runtime.email}"
}

locals {
  # Common Cloud Run scaling/cost knobs reused by every service.
  scaling_min = var.min_instances
  scaling_max = var.max_instances
}

# ---- MCP tool server (standalone; agents reference its URL) ----
resource "google_cloud_run_v2_service" "mcp" {
  name                = "ts-mcp"
  location            = var.region
  deletion_protection = false
  ingress             = "INGRESS_TRAFFIC_ALL"
  template {
    service_account = google_service_account.runtime.email
    scaling {
      min_instance_count = local.scaling_min
      max_instance_count = local.scaling_max
    }
    timeout = "300s"
    containers {
      image = var.image
      ports {
        container_port = 8080
      }
      env {
        name  = "ROLE"
        value = "mcp"
      }
      env {
        name  = "DATA_PATH"
        value = "data.csv"
      }
      dynamic "env" {
        for_each = local.vertex_env
        content {
          name  = env.key
          value = env.value
        }
      }
      resources {
        limits = { cpu = "1", memory = "2Gi" }
      }
    }
  }
  depends_on = [google_project_service.apis, google_project_iam_member.vertex_user]
}

# ---- Predictor / explainer agents (consume MCP tools) ----
resource "google_cloud_run_v2_service" "agent" {
  for_each = {
    "ts-zero-shot"  = "zero_shot"
    "ts-fine-tuned" = "fine_tuned"
    "ts-explainer"  = "explainer"
  }
  name                = each.key
  location            = var.region
  deletion_protection = false
  ingress             = "INGRESS_TRAFFIC_ALL"
  template {
    service_account = google_service_account.runtime.email
    scaling {
      min_instance_count = local.scaling_min
      max_instance_count = local.scaling_max
    }
    timeout = "300s"
    containers {
      image = var.image
      ports {
        container_port = 8080
      }
      env {
        name  = "ROLE"
        value = each.value
      }
      env {
        name  = "MCP_URL"
        value = local.mcp_url
      }
      dynamic "env" {
        for_each = local.vertex_env
        content {
          name  = env.key
          value = env.value
        }
      }
      resources {
        limits = { cpu = "1", memory = "2Gi" }
      }
    }
  }
  depends_on = [google_cloud_run_v2_service.mcp]
}

# ---- Orchestrator (A2A entry point; references the real agent URLs) ----
resource "google_cloud_run_v2_service" "orchestrator" {
  name                = "ts-orchestrator"
  location            = var.region
  deletion_protection = false
  ingress             = "INGRESS_TRAFFIC_ALL"
  template {
    service_account = google_service_account.runtime.email
    scaling {
      min_instance_count = local.scaling_min
      max_instance_count = local.scaling_max
    }
    timeout = "300s"
    containers {
      image = var.image
      ports {
        container_port = 8080
      }
      env {
        name  = "ROLE"
        value = "orchestrator"
      }
      env {
        name  = "MCP_URL"
        value = local.mcp_url
      }
      env {
        name  = "ZERO_SHOT_URL"
        value = google_cloud_run_v2_service.agent["ts-zero-shot"].uri
      }
      env {
        name  = "FINE_TUNED_URL"
        value = google_cloud_run_v2_service.agent["ts-fine-tuned"].uri
      }
      env {
        name  = "EXPLAINER_URL"
        value = google_cloud_run_v2_service.agent["ts-explainer"].uri
      }
      env {
        name  = "RECONCILE"
        value = var.reconcile
      }
      dynamic "env" {
        for_each = local.vertex_env
        content {
          name  = env.key
          value = env.value
        }
      }
      env {
        name = "API_AUTH_TOKEN"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.auth.secret_id
            version = "latest"
          }
        }
      }
      resources {
        limits = { cpu = "1", memory = "2Gi" }
      }
    }
  }
  depends_on = [google_cloud_run_v2_service.agent, google_secret_manager_secret_version.auth]
}

# ---- Browser demo UI (optional; proxies to the orchestrator) ----
# A public web page so non-technical users can try the service without curl.
# It holds the bearer token server-side and forwards to /verify, so the token is
# never exposed to the browser. Set var.enable_demo = false to skip it.
resource "google_cloud_run_v2_service" "demo" {
  count               = var.enable_demo ? 1 : 0
  name                = "ts-demo"
  location            = var.region
  deletion_protection = false
  ingress             = "INGRESS_TRAFFIC_ALL"
  template {
    service_account = google_service_account.runtime.email
    scaling {
      min_instance_count = local.scaling_min
      max_instance_count = local.scaling_max
    }
    timeout = "300s"
    containers {
      image = var.image
      ports {
        container_port = 8080
      }
      env {
        name  = "ROLE"
        value = "demo"
      }
      env {
        name  = "ORCHESTRATOR_URL"
        value = google_cloud_run_v2_service.orchestrator.uri
      }
      env {
        name = "API_AUTH_TOKEN"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.auth.secret_id
            version = "latest"
          }
        }
      }
      resources {
        limits = { cpu = "1", memory = "512Mi" }
      }
    }
  }
  depends_on = [google_cloud_run_v2_service.orchestrator]
}

# Allow invocation. The orchestrator is additionally protected by the bearer
# token at the application layer; internal agents/MCP are reachable by URL only.
# (Hardening option: require IAM auth on the internal services and have the
# orchestrator attach a Google ID token — see deploy README.)
resource "google_cloud_run_v2_service_iam_member" "public" {
  for_each = toset(concat([
    google_cloud_run_v2_service.mcp.name,
    google_cloud_run_v2_service.agent["ts-zero-shot"].name,
    google_cloud_run_v2_service.agent["ts-fine-tuned"].name,
    google_cloud_run_v2_service.agent["ts-explainer"].name,
    google_cloud_run_v2_service.orchestrator.name,
  ], var.enable_demo ? [google_cloud_run_v2_service.demo[0].name] : []))
  name     = each.value
  location = var.region
  role     = "roles/run.invoker"
  member   = "allUsers"
}
