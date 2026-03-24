# Load Benchmarking

This directory provides a Locust load test suite for `reporium-api`.

## Prerequisites

- Python 3.11+
- Locust installed:

```bash
pip install -r requirements-dev.txt
```

## Running the Load Tests

```bash
locust -f tests/load/locustfile.py --host https://reporium-api-573778300586.us-central1.run.app
```

This opens the Locust web UI at `http://localhost:8089` where you can configure user count and spawn rate interactively.

### Headless (CI / scripted) runs

```bash
locust -f tests/load/locustfile.py \
  --host https://reporium-api-573778300586.us-central1.run.app \
  --headless -u 10 -r 2 -t 2m
```

## Endpoints Covered

| Method | Path | Weight | Notes |
|--------|------|--------|-------|
| `GET` | `/library/full?page=1&pageSize=200` | 3 | Most common read |
| `GET` | `/library` | 2 | |
| `GET` | `/stats` | 2 | |
| `POST` | `/intelligence/ask` | 1 | Body: `{"question": "what AI tools do you use for testing?"}` |
| `GET` | `/health` | 1 | |

## User Configuration

- `wait_time = between(1, 3)` seconds between tasks per simulated user
- Tasks use `@task(weight)` decorators to control relative frequency

## Metrics To Watch

- p50, p95, and p99 latency per endpoint
- Failure rate
- Request throughput (RPS)

## Proposed SLOs

- p95 `GET /library/full` < 2s (when cached)
- p95 `POST /intelligence/ask` < 15s
- p99 error rate < 1%
