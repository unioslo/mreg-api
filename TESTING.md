# Testing

mreg-api has two types of tests:

- **Unit tests** — Unit testing of functions and methods used by the package. No interaction with a live mreg server is required.
- **Integration tests** — Run against a live (containerized) mreg server. Skipped automatically when no server URL is configured.

## Unit tests

Unit tests use `pytest-httpserver` to mock MREG server responses for any tests that require HTTP interactions. No external services required.

Bare `pytest` invocations ignore integration tests (skipped automatically when no server URL is configured):

```bash
uv run pytest
```

In case some configuration defines `MREG_URL`, unit tests can still be run exclusively by ignoring the integration tests via markers:

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
MREG_DOMAIN=example.com \
MREG_CACHE=0 \
MREG_TEST_NETWORK=10.0.0.0/8 \
  uv run pytest tests/integration/ -v

# Via CLI flags instead of env vars
uv run pytest tests/integration/ \
  --mreg-url http://127.0.0.1:8000 \
  --mreg-username myuser \
  --mreg-password secret
```

### Configuration

| Env var | CLI flag | Default | Description |
|---|---|---|---|
| `MREG_URL` | `--mreg-url` | *(none — required to activate integration tests)* | mreg server base URL |
| `MREG_USERNAME` | `--mreg-username` | `test` | Login username |
| `MREG_PASSWORD` | `--mreg-password` | `test` | Login password |
| `MREG_DOMAIN` | `--mreg-domain` | `example.com` | Default domain for the client |
| `MREG_CACHE` | `--mreg-cache` | `false` | Enable mreg client cache (`1`/`true`/`yes` = on) |
| `MREG_TEST_NETWORK` | `--test-network` | `10.0.0.0/8` | Network CIDR created as shared test network |
| `MREG_TEST_IP` | `--test-ip` | `10.0.0.1` | IP address used in IP-related tests |

CLI flags take precedence over env vars.

### Seed data

Tests that depend on a zone (`example.com`) or bulk DHCP data require `ci/seed.py` to have run first:

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
