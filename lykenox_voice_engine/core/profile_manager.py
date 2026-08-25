"""Profile persistence."""

import json
from pathlib import Path

from lykenox_voice_engine.config.settings import PROJECT_ROOT
from lykenox_voice_engine.models.profile import VoiceProfile


class ProfileManager:
    """Load and save local voice profiles."""

    def __init__(self, profiles_dir: Path | None = None) -> None:
        """Create the manager and ensure the LYKENOX profile exists."""
        self.profiles_dir = profiles_dir or PROJECT_ROOT / "profiles"
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self.ensure_default_profile()

    def ensure_default_profile(self) -> VoiceProfile:
        """Create LYKENOX Voice profile metadata if absent."""
        profile = VoiceProfile()
        path = self.profile_path(profile.id)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(profile.to_dict(), indent=2), encoding="utf-8")
        return self.get_profile(profile.id)

    def profile_path(self, profile_id: str) -> Path:
        """Return the profile JSON path."""
        return self.profiles_dir / profile_id / "profile.json"

    def list_profiles(self) -> list[VoiceProfile]:
        """Return all known profiles."""
        profiles = []
        for path in self.profiles_dir.glob("*/profile.json"):
            profiles.append(VoiceProfile.from_dict(json.loads(path.read_text(encoding="utf-8"))))
        return profiles

    def get_profile(self, profile_id: str) -> VoiceProfile:
        """Return one profile by id."""
        path = self.profile_path(profile_id)
        if not path.exists():
            raise FileNotFoundError(f"Perfil no encontrado: {profile_id}")
        return VoiceProfile.from_dict(json.loads(path.read_text(encoding="utf-8")))
