# Testing

## Unit tests

Unit tests use `pytest-httpserver` to spin up a local HTTP server. No external services required.

```bash
uv run pytest tests/ --ignore=tests/integration
```

Or by marker:

```bash
uv run pytest -m "not integration"
```

## Integration tests

Integration tests run against a live mreg server. They are **skipped automatically** when no server URL is configured, so `uv run pytest` never fails due to a missing server.

### Prerequisites

A running mreg instance (and PostgreSQL). The fastest way is the containerized setup:

```bash
bash ci/run_tests.sh

# If port 8000 is already in use (e.g., a local mreg dev server):
MREG_PORT=8081 bash ci/run_tests.sh
```

This starts containers, creates a superuser, runs optional seed data, then runs both unit and integration tests with combined coverage output in `htmlcov/`.

### Running manually against a local server

```bash
# Minimal — username/password default to "test"
MREG_URL=http://127.0.0.1:8000 uv run pytest tests/integration/

# All options explicit
MREG_URL=http://127.0.0.1:8000 \
MREG_USERNAME=myuser \
MREG_PASSWORD=secret \
MREG_DOMAIN=uio.no \
MREG_TEST_NETWORK=10.0.0.0/8 \
  uv run pytest tests/integration/ -v

# Via CLI flags instead of env vars
uv run pytest tests/integration/ \
  --mreg-url http://127.0.0.1:8000 \
  --mreg-username myuser \
  --mreg-password secret
```

### Seed data

Tests that depend on a zone (`example.uio.no`) or bulk DHCP data require `ci/seed.py` to have run first:

```bash
MREG_URL=http://127.0.0.1:8000 MREG_USERNAME=test MREG_PASSWORD=test \
  uv run python ci/seed.py
```

The seed script reads `ci/seed_data.yaml`. Tests that cannot find their prerequisite zone call `pytest.skip()` automatically.

### Subset runs

```bash
# Read-only tests only (safe against staging/production)
uv run pytest tests/integration/ -m readonly

# Single resource type
uv run pytest tests/integration/test_labels.py

# Specific test
uv run pytest tests/integration/test_labels.py::test_delete_by_name
```

## Coverage

### Unit tests only

```bash
uv run pytest tests/ --ignore=tests/integration \
  --cov=mreg_api --cov-report=term-missing
```

### Combined unit + integration

Run `ci/run_tests.sh` (uses `--cov-append` to merge both passes), or manually:

```bash
# Pass 1: unit tests — write .coverage
uv run pytest tests/ --ignore=tests/integration \
  --cov=mreg_api --cov-report=

# Pass 2: integration tests — append to .coverage
MREG_URL=http://127.0.0.1:8000 MREG_USERNAME=test MREG_PASSWORD=test \
  uv run pytest tests/integration/ \
  --cov=mreg_api --cov-append --cov-report=html --cov-report=term

# Open report
open htmlcov/index.html
```

## Markers

| Marker | Meaning |
|---|---|
| `integration` | Requires live mreg server — skipped without `MREG_URL` |
| `readonly` | Test only reads data, safe against staging/production servers |
