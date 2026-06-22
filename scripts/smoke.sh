#!/usr/bin/env bash
# Reporium $0 local-OSS substrate smoke test (REPORIUM-$0-01, issue #6).
#
# Proves the substrate runs $0/local end-to-end:
#   1. docker compose up --build --wait   (all services report healthy)
#   2. assert each OSS substitute is reachable (Postgres, Redis, Pub/Sub, API)
#   3. assert the API /health reports db=ok (migrations applied, real DB query)
#   4. assert the Pub/Sub emulator pre-created the reporium-events topic
#   5. docker compose down -v             (clean teardown, volumes removed)
#
# No cloud, no secrets, no paid services. Exit 0 = PASS, non-zero = FAIL.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
COMPOSE="docker compose -f local/docker-compose.yml"

pass() { printf '  [PASS] %s\n' "$1"; }
fail() { printf '  [FAIL] %s\n' "$1"; FAILED=1; }
FAILED=0

cleanup() {
  echo "--- teardown (down -v) ---"
  $COMPOSE down -v || true
}
trap cleanup EXIT

echo "=== Reporium local substrate smoke test ($$0/local) ==="

echo "--- bringing stack up (build + wait healthy) ---"
$COMPOSE up -d --build --wait

echo "--- service health ---"
for svc in db redis pubsub api; do
  state="$($COMPOSE ps "$svc" --format '{{.Health}}' 2>/dev/null || true)"
  if [ "$state" = "healthy" ]; then pass "$svc healthy"; else fail "$svc not healthy (state='$state')"; fi
done

echo "--- API /health (db-backed) ---"
body="$(curl -fsS --max-time 10 http://localhost:8080/health || true)"
echo "    response: $body"
case "$body" in
  *'"status":"ok"'*'"db":"ok"'*|*'"db":"ok"'*'"status":"ok"'*) pass "API /health status=ok db=ok" ;;
  *) fail "API /health did not report ok/db=ok" ;;
esac

echo "--- Postgres pgvector extension reachable ---"
if $COMPOSE exec -T db psql -U postgres -d reporium -tAc "SELECT 1" | grep -q 1; then
  pass "Postgres query SELECT 1"
else
  fail "Postgres query failed"
fi

echo "--- Redis PING ---"
if [ "$($COMPOSE exec -T redis redis-cli ping | tr -d '\r')" = "PONG" ]; then
  pass "Redis PONG"
else
  fail "Redis did not PONG"
fi

echo "--- Pub/Sub emulator topic (local broker, replaces GCP Pub/Sub) ---"
# The emulator's REST surface lists pre-created topics for the project.
topics="$(curl -fsS --max-time 10 \
  http://localhost:8681/v1/projects/reporium-local/topics || true)"
echo "    topics: $topics"
if echo "$topics" | grep -q "reporium-events"; then
  pass "Pub/Sub emulator has topic reporium-events"
else
  fail "Pub/Sub emulator missing reporium-events topic"
fi

echo ""
if [ "$FAILED" = "0" ]; then
  echo "=== SMOKE PASS — substrate runs \$0/local ==="
else
  echo "=== SMOKE FAIL ==="
fi
exit "$FAILED"
