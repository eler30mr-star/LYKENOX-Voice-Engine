"""Small Python client for the local LYKENOX Voice Engine API."""

from __future__ import annotations

import requests


class LykenoxVoiceClient:
    """Convenience client for local singing synthesis jobs."""

    def __init__(self, base_url: str = "http://127.0.0.1:8765") -> None:
        self.base_url = base_url.rstrip("/")

    def health(self) -> dict:
        """Return API health state."""

        return requests.get(f"{self.base_url}/health", timeout=10).json()

    def profiles(self) -> list[dict]:
        """Return available voice profiles."""

        return requests.get(f"{self.base_url}/profiles", timeout=10).json()

    def synthesize(self, payload: dict) -> dict:
        """Submit a lyrics plus notes synthesis request."""

        response = requests.post(f"{self.base_url}/synthesize", json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
