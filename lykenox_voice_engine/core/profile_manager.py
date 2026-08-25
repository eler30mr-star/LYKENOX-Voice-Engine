"""Profile discovery and loading."""

from __future__ import annotations

from pathlib import Path

from lykenox_voice_engine.models.profile import VoiceProfile


class ProfileManager:
    """Manage local voice profile folders."""

    def __init__(self, profiles_dir: Path) -> None:
        self.profiles_dir = profiles_dir

    def list_profiles(self) -> list[VoiceProfile]:
        """Return all profiles containing a profile.json file."""

        return [VoiceProfile.load(path) for path in sorted(self.profiles_dir.glob("*/profile.json"))]

    def get_profile(self, profile_id: str) -> VoiceProfile:
        """Load one profile by id or raise FileNotFoundError."""

        path = self.profiles_dir / profile_id / "profile.json"
        if not path.exists():
            raise FileNotFoundError(f"profile not found: {profile_id}")
        return VoiceProfile.load(path)
