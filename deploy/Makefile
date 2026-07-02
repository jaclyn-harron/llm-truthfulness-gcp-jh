# Deployment helpers. Infra is declarative Terraform; the only imperative step
# is building the image (one Cloud Build command).
#
# Usage:
#   export TF_VAR_project_id=my-project TF_VAR_region=us-central1
#   export TF_VAR_fine_tuned_model=projects/.../locations/.../endpoints/...
#   export TF_VAR_api_auth_token=$(openssl rand -hex 24)
#   make deploy        # repo + image + all services, end to end
#
PROJECT ?= $(TF_VAR_project_id)
REGION  ?= $(or $(TF_VAR_region),us-central1)
REPO    := $(REGION)-docker.pkg.dev/$(PROJECT)/truthfulness
TAG     ?= latest
IMAGE   := $(REPO)/app:$(TAG)

.PHONY: compose-up bootstrap push tf-init deploy destroy

compose-up:
	cd .. && docker compose -f deploy/docker-compose.yml up --build

# Create just the Artifact Registry repo (and enable APIs) so we can push.
bootstrap: tf-init
	cd terraform && terraform apply -auto-approve \
	  -target=google_project_service.apis -target=google_artifact_registry_repository.repo \
	  -var="image=$(IMAGE)"

push:
	cd .. && gcloud builds submit --config cloudbuild.yaml \
	  --substitutions=_IMAGE=$(IMAGE) --project $(PROJECT) .

tf-init:
	cd terraform && terraform init

# Full deploy: bootstrap registry -> build/push image -> apply everything.
deploy: bootstrap push
	cd terraform && terraform apply -auto-approve -var="image=$(IMAGE)"

destroy:
	cd terraform && terraform destroy -var="image=$(IMAGE)"
