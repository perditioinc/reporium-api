# Load Benchmarking

This directory provides a lightweight Locust harness for recurring load and concurrency checks against `reporium-api`.

## Prerequisites

- Python 3.11+
- repo dependencies installed
- Locust installed locally:

```bash
pip install locust
```

- API target available locally or in staging

## Files

- `tests/load/locustfile.py`: baseline Locust workload for `/library/full` and `/intelligence/ask`

## Endpoints Covered

- `GET /library/full`
- `POST /intelligence/ask`

### `/intelligence/ask` payload

Default request body:

```json
{
  "question": "What repositories help with AI evaluation and observability?",
  "top_k": 8
}
```

Override with environment variables:

- `LOAD_INTELLIGENCE_QUESTION`
- `LOAD_INTELLIGENCE_TOP_K`
- `LOAD_INTELLIGENCE_ASK_PATH`

If you want to exercise an authenticated variant instead, set:

- `LOAD_INTELLIGENCE_ASK_PATH=/intelligence/query`
- `LOAD_TEST_BEARER_TOKEN=<token>`

## Local Runs

Set the target host:

```bash
$env:LOCUST_HOST="http://localhost:8000"
```

### Baseline sustained

Target: approximately 1 request/second overall.

```bash
locust -f tests/load/locustfile.py --host $env:LOCUST_HOST --headless -u 1 -r 1 -t 5m
```

### Burst

Target: 10 users ramped immediately for 60 seconds.

```bash
locust -f tests/load/locustfile.py --host $env:LOCUST_HOST --headless -u 10 -r 10 -t 60s
```

## Staging Runs

```bash
$env:LOCUST_HOST="https://your-staging-host"
locust -f tests/load/locustfile.py --host $env:LOCUST_HOST --headless -u 10 -r 10 -t 60s
```

If staging requires an authenticated intelligence endpoint:

```bash
$env:LOAD_INTELLIGENCE_ASK_PATH="/intelligence/query"
$env:LOAD_TEST_BEARER_TOKEN="replace-me"
locust -f tests/load/locustfile.py --host $env:LOCUST_HOST --headless -u 10 -r 10 -t 60s
```

## Metrics To Watch

- p50, p95, and p99 latency
- failure rate per endpoint
- request throughput
- saturation of `/library/full` cache behavior
- downstream error responses from `/intelligence/ask`

## Proposed SLOs

- p95 `GET /library/full` < 2s when cached
- p95 `POST /intelligence/ask` < 15s
- p99 error rate < 1%

## Monthly Run Guidance

- run the baseline profile monthly
- run the burst profile before major releases and after infra changes
- archive the command, target environment, and key latency/error metrics with the run notes
