"""Tests for profile loading."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from lykenox_voice_engine.core.profile_manager import ProfileManager


class TestProfileManager(unittest.TestCase):
    """Check profile discovery behavior."""

    def test_lists_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            profile_dir = Path(temp_dir) / "lykenox"
            profile_dir.mkdir()
            (profile_dir / "profile.json").write_text(
                '{"id":"lykenox","name":"LYKENOX Voice","dataset_duration":0,'
                '"clips":0,"model_type":"none","sample_rate":44100,'
                '"training_steps":0,"training_epochs":0,"checkpoint":null,'
                '"speaker_embedding":null,"status":"test"}',
                encoding="utf-8",
            )
            profiles = ProfileManager(Path(temp_dir)).list_profiles()
            self.assertEqual(profiles[0].name, "LYKENOX Voice")


if __name__ == "__main__":
    unittest.main()
