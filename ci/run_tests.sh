#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

RUN_UNIT=true
RUN_INTEGRATION=true
RUN_COVERAGE=true
MREG_IMAGE="${MREG_IMAGE:-ghcr.io/unioslo/mreg:master}"
MREG_IMAGE_PULL_POLICY="${MREG_IMAGE_PULL_POLICY:-missing}"
# Host port the mreg container is exposed on. Override if 8000 is already in use.
MREG_PORT="${MREG_PORT:-8000}"
MREG_URL="${MREG_URL:-http://127.0.0.1:${MREG_PORT}}"
MREG_USERNAME="${MREG_USERNAME:-test}"
MREG_PASSWORD="${MREG_PASSWORD:-test}"
MREG_CACHE="${MREG_CACHE:-0}"

usage() {
    cat <<'EOF'
Usage: bash ci/run_tests.sh [--unit-only|--integration-only] [--no-cov] [-- pytest args...]

Everything after -- is passed to pytest:

  bash ci/run_tests.sh --integration-only -- tests/integration/test_labels.py
  bash ci/run_tests.sh --integration-only -- -x -k labels
EOF
}

# Extra arguments passed to pytest (everything after --).
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --unit-only)        RUN_INTEGRATION=false; shift ;;
        --integration-only) RUN_UNIT=false; shift ;;
        --no-cov)           RUN_COVERAGE=false; shift ;;
        -h|--help)          usage; exit 0 ;;
        --)                 shift; EXTRA_ARGS=("$@"); break ;;
        *)
            echo "ERROR: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if command -v podman &>/dev/null && ! command -v docker &>/dev/null; then
    DOCKER=podman
else
    DOCKER=docker
fi

if [[ "$RUN_INTEGRATION" == "true" ]]; then
    # Check if port is already in use before starting containers
    if lsof -i ":${MREG_PORT}" 2>/dev/null | grep -q LISTEN; then
        echo "ERROR: Port ${MREG_PORT} is already in use." >&2
        echo "Set MREG_PORT to a free port: MREG_PORT=8081 bash ci/run_tests.sh" >&2
        exit 1
    fi

    cleanup() {
        cd "$SCRIPT_DIR"
        $DOCKER compose down --remove-orphans 2>/dev/null || true
    }
    trap cleanup EXIT

    cd "$SCRIPT_DIR"

    echo "Starting mreg and postgres on port ${MREG_PORT}..."
    MREG_IMAGE="$MREG_IMAGE" MREG_PORT="$MREG_PORT" \
    MREG_IMAGE_PULL_POLICY="$MREG_IMAGE_PULL_POLICY" \
        $DOCKER compose up -d

    echo "Creating superuser..."
    $DOCKER exec mreg uv run /app/manage.py create_mreg_superuser \
        --username "$MREG_USERNAME" --password "$MREG_PASSWORD" 2>/dev/null \
        || $DOCKER exec mreg /app/manage.py create_mreg_superuser \
            --username "$MREG_USERNAME" --password "$MREG_PASSWORD"


fi

cd "$REPO_ROOT"

UNIT_EXIT=0
INTEGRATION_EXIT=0

if [[ "$RUN_UNIT" == "true" ]]; then
    echo "Running unit tests..."
    # No path here: pytest falls back to testpaths (tests/) from pyproject.toml,
    # so a path in EXTRA_ARGS narrows the run instead of being added to it.
    UNIT_ARGS=(-m "not integration" -v)
    if [[ "$RUN_COVERAGE" == "true" ]]; then
        UNIT_ARGS+=(--cov=mreg_api --cov-report=)
    else
        UNIT_ARGS+=(--no-cov)
    fi
    UNIT_ARGS+=(${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"})
    uv run pytest "${UNIT_ARGS[@]}" || UNIT_EXIT=$?
fi

if [[ "$RUN_INTEGRATION" == "true" ]]; then
    echo "Running integration tests..."
    INTEGRATION_ARGS=(-m integration -v)
    if [[ "$RUN_COVERAGE" == "true" ]]; then
        INTEGRATION_ARGS+=(--cov=mreg_api --cov-append --cov-report=html --cov-report=term)
    else
        INTEGRATION_ARGS+=(--no-cov)
    fi
    INTEGRATION_ARGS+=(${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"})
    MREG_URL="$MREG_URL" \
    MREG_USERNAME="$MREG_USERNAME" \
    MREG_PASSWORD="$MREG_PASSWORD" \
        uv run pytest "${INTEGRATION_ARGS[@]}" || INTEGRATION_EXIT=$?
fi

if [[ $UNIT_EXIT -ne 0 || $INTEGRATION_EXIT -ne 0 ]]; then
    exit 1
fi
exit 0
