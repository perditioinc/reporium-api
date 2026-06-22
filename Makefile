# Root passthrough to the $0 local-OSS substrate (REPORIUM-$0-01, issue #6).
# Lets you run `make up | down | smoke | seed | logs | ps` from the repo root.
# All real targets live in local/Makefile.

.PHONY: up down smoke seed logs ps rebuild help
up down smoke seed logs ps rebuild help:
	@$(MAKE) -f local/Makefile $@
