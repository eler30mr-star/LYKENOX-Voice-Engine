"""Small Python client for the local LYKENOX Voice Engine API."""

import json
from urllib import request as urlrequest


class LykenoxVoiceClient:
    """HTTP client for local singing synthesis jobs."""

    def __init__(self, base_url: str = "http://127.0.0.1:8765") -> None:
        """Create a client for the local-only API."""
        self.base_url = base_url.rstrip("/")

    def synthesize(self, profile: str, lyrics: str, notes: list[dict], tempo: int) -> dict:
        """Submit a score-to-singing request."""
        payload = json.dumps({
            "profile": profile,
            "lyrics": lyrics,
            "notes": notes,
            "tempo": tempo,
            "output_format": "wav",
        }).encode("utf-8")
        req = urlrequest.Request(
            f"{self.base_url}/synthesize",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlrequest.urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
