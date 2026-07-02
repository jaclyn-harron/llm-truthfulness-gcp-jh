output "orchestrator_url" {
  value       = google_cloud_run_v2_service.orchestrator.uri
  description = "Public entry point. POST {orchestrator_url}/verify with the bearer token."
}

output "verify_endpoint" {
  value = "${google_cloud_run_v2_service.orchestrator.uri}/verify"
}

output "agent_urls" {
  value = {
    mcp        = google_cloud_run_v2_service.mcp.uri
    zero_shot  = google_cloud_run_v2_service.agent["ts-zero-shot"].uri
    fine_tuned = google_cloud_run_v2_service.agent["ts-fine-tuned"].uri
    explainer  = google_cloud_run_v2_service.agent["ts-explainer"].uri
  }
}

output "demo_url" {
  value       = var.enable_demo ? google_cloud_run_v2_service.demo[0].uri : null
  description = "Public browser demo UI (no token needed by the visitor)."
}

output "artifact_registry_repo" {
  value = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.repo.repository_id}"
}
