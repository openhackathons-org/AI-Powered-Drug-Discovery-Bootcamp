# Copyright (c) 2026, NVIDIA CORPORATION. Licensed under the Apache License, Version 2.0 (the "License") you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

"""Helpers for reading generated AI-Powered-Drug-Discovery-Bootcamp NIM endpoint settings."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List


def _candidate_env_files() -> Iterable[Path]:
    explicit = os.environ.get("OPENHACKATHON_ENV_FILE")
    if explicit:
        yield Path(explicit)

    cwd = Path.cwd().resolve()
    for path in (cwd, *cwd.parents):
        yield path / ".openhackathon-nims.env"

    yield Path(__file__).resolve().parents[1] / ".openhackathon-nims.env"


def load_openhackathon_env() -> None:
    """Load .openhackathon-nims.env into os.environ without overriding values."""
    for env_file in _candidate_env_files():
        if not env_file.exists():
            continue

        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
        return


def normalize_endpoint(value: str, base_url: str = "http://localhost") -> str:
    value = value.strip()
    if not value:
        return ""
    if "|" in value:
        value = value.split("|", 1)[0].strip()
    if value.startswith("http://") or value.startswith("https://"):
        return value.rstrip("/")
    if value.isdigit():
        return f"http://localhost:{value}"
    if ":" in value and "/" not in value:
        return f"http://{value}"

    return base_url.rstrip("/")


def normalize_endpoint_urls(value: str, base_url: str = "http://localhost") -> List[str]:
    return [
        endpoint
        for endpoint in (normalize_endpoint(part, base_url) for part in value.split(","))
        if endpoint
    ]


def boltz2_endpoint_urls(default: str = "8000") -> List[str]:
    load_openhackathon_env()
    endpoints = os.environ.get("BOLTZ2_ENDPOINTS") or os.environ.get("BOLTZ2_URL") or default
    return normalize_endpoint_urls(endpoints, os.environ.get("BOLTZ2_URL", "http://localhost"))
