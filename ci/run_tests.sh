#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

RUN_UNIT=true
RUN_INTEGRATION=true
MREG_IMAGE="${MREG_IMAGE:-ghcr.io/unioslo/mreg:master}"
# Host port the mreg container is exposed on. Override if 8000 is already in use.
MREG_PORT="${MREG_PORT:-8000}"
MREG_URL="${MREG_URL:-http://127.0.0.1:${MREG_PORT}}"
MREG_USERNAME="${MREG_USERNAME:-test}"
MREG_PASSWORD="${MREG_PASSWORD:-test}"
MREG_CACHE="${MREG_CACHE:-0}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --unit-only)        RUN_INTEGRATION=false; shift ;;
        --integration-only) RUN_UNIT=false; shift ;;
        *) shift ;;
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
    MREG_IMAGE="$MREG_IMAGE" MREG_PORT="$MREG_PORT" $DOCKER compose up -d

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
    uv run pytest tests/ --ignore=tests/integration \
        --cov=mreg_api --cov-report= -q \
        -v
    UNIT_EXIT=$?
fi

if [[ "$RUN_INTEGRATION" == "true" ]]; then
    echo "Running integration tests..."
    MREG_URL="$MREG_URL" \
    MREG_USERNAME="$MREG_USERNAME" \
    MREG_PASSWORD="$MREG_PASSWORD" \
        uv run pytest tests/integration/ \
            --cov=mreg_api --cov-append \
            --cov-report=html --cov-report=term \
            -v
    INTEGRATION_EXIT=$?
fi

if [[ $UNIT_EXIT -ne 0 || $INTEGRATION_EXIT -ne 0 ]]; then
    exit 1
fi
exit 0
