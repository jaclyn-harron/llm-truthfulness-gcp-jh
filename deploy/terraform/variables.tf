variable "project_id" {
  type        = string
  description = "GCP project ID to deploy into."
}

variable "region" {
  type        = string
  default     = "us-central1"
  description = "Cloud Run region."
}

variable "vertex_location" {
  type        = string
  default     = ""
  description = "Vertex AI location for model calls. Defaults to var.region. Must match the region the tuned model endpoint lives in."
}

variable "image" {
  type        = string
  description = "Full container image URL (e.g. REGION-docker.pkg.dev/PROJECT/truthfulness/app:TAG). Build & push with `make push` first."
}

variable "fine_tuned_model" {
  type        = string
  description = "Tuned-model endpoint resource name (projects/…/locations/…/endpoints/…) produced by the Vertex supervised tuning job."
}

variable "api_auth_token" {
  type        = string
  sensitive   = true
  description = "Bearer token clients must send to the public orchestrator /verify endpoint."
}

variable "reconcile" {
  type        = string
  default     = "fine_tuned"
  description = "Disagreement strategy: fine_tuned | confidence | zero_shot."
}

variable "min_instances" {
  type        = number
  default     = 0
  description = "Min Cloud Run instances per service. 0 = scale to zero (cheapest)."
}

variable "max_instances" {
  type        = number
  default     = 2
  description = "Max Cloud Run instances per service (cost ceiling)."
}

variable "enable_demo" {
  type        = bool
  default     = true
  description = "Deploy a public browser demo UI that proxies to the orchestrator."
}
