# Reporium $0 local-OSS substrate (REPORIUM-$0-01, issue #6)
#
# All targets run the OSS local stack with ZERO cloud spend. No GCP, no paid
# services, no secrets. Run from the repo root:  make -f local/Makefile <target>
# (a thin root passthrough is provided so `make up` also works.)

COMPOSE := docker compose -f local/docker-compose.yml
SMOKE   := bash scripts/smoke.sh

.PHONY: help up down logs ps seed smoke rebuild

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

up: ## Build + start the full local substrate, wait until healthy
	$(COMPOSE) up -d --build --wait
	@echo ""
	@echo "Reporium local substrate is UP ($$0/local):"
	@echo "  API     http://localhost:8080/health   (docs: /docs)"
	@echo "  Postgres localhost:5432  (postgres/postgres, db=reporium)"
	@echo "  Redis    localhost:6379"
	@echo "  Pub/Sub  localhost:8681  (emulator, project=reporium-local)"

down: ## Stop the stack and DELETE volumes (clean teardown)
	$(COMPOSE) down -v

logs: ## Tail logs from all services
	$(COMPOSE) logs -f

ps: ## Show service status
	$(COMPOSE) ps

seed: ## Apply DB migrations (idempotent; api `up` already runs this)
	$(COMPOSE) run --rm migrate

smoke: ## Bring up, health-check every service, then tear down -v
	$(SMOKE)

rebuild: ## Force a clean rebuild of the api image
	$(COMPOSE) build --no-cache api migrate
