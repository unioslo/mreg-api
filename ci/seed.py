"""Seed a containerized mreg instance with baseline data from seed_data.yaml.

Run via ci/run_tests.sh or directly:
    MREG_URL=http://127.0.0.1:8000 MREG_USERNAME=test MREG_PASSWORD=test uv run python ci/seed.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from mreg_api.client import MregClient
from mreg_api.exceptions import EntityAlreadyExists
from mreg_api.exceptions import PostError

SEED_FILE = Path(__file__).parent / "seed_data.yaml"


def main() -> None:
    url = os.environ.get("MREG_URL", "http://127.0.0.1:8000")
    username = os.environ.get("MREG_USERNAME", "test")
    password = os.environ.get("MREG_PASSWORD", "test")
    domain = os.environ.get("MREG_DOMAIN", "example.com")

    if not SEED_FILE.exists():
        print(f"No seed file found at {SEED_FILE}, skipping.")
        return

    data = yaml.safe_load(SEED_FILE.read_text())

    client = MregClient(url=url, domain=domain, cache=False)
    client.login(username=username, password=password)

    for zone_spec in data.get("zones", []):
        name = zone_spec["name"]
        try:
            client.zone.create(
                name=name,
                email=zone_spec["email"],
                primary_ns=zone_spec["primary_ns"],
                force=True,
            )
            print(f"Created zone: {name}")
        except (EntityAlreadyExists, PostError):
            print(f"Zone already exists (skipping): {name}")

    for net_spec in data.get("networks", []):
        network = net_spec["network"]
        existing = client.network.get(network, required=False)
        if existing is not None:
            print(f"Network already exists (skipping): {network}")
            continue
        try:
            client.network.create(
                network=network,
                description=net_spec.get("description", ""),
            )
            print(f"Created network: {network}")
        except PostError as e:
            if "overlaps" in str(e) or "already exists" in str(e).lower():
                print(f"Network overlaps/exists (skipping): {network}")
            else:
                raise

    print("Seed complete.")


if __name__ == "__main__":
    main()
