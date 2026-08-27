from pathlib import Path

import torch

from lykenox_voice_engine.models.vocoder import (
    LykenoxMultiScaleWaveformDiscriminator,
    LykenoxVocoderGeneratorV41,
    VOCODER_GENERATOR_V4_1_ARCHITECTURE,
)
from lykenox_voice_engine.training.speech_pitch import PITCH_TARGET_VERSION
from lykenox_voice_engine.training.speech_vocoder_data import VOCODER_SEGMENT_CONTRACT_VERSION
from lykenox_voice_engine.training.speech_vocoder_losses import VOCODER_LOSS_RECIPE_VERSION
from lykenox_voice_engine.training.speech_vocoder_source_balance import VOCODER_SOURCE_BALANCE_VERSION
from lykenox_voice_engine.training.speech_vocoder_source_filter_artifact import (
    load_source_filter_checkpoint,
    save_source_filter_checkpoint,
)


def _provenance() -> dict[str, object]:
    return {
        "segment_contract_version": VOCODER_SEGMENT_CONTRACT_VERSION,
        "loss_recipe_version": VOCODER_LOSS_RECIPE_VERSION,
        "pitch_target_version": PITCH_TARGET_VERSION,
        "source_balance_version": VOCODER_SOURCE_BALANCE_VERSION,
    }


def test_v41_checkpoint_roundtrip_exact(tmp_path: Path) -> None:
    torch.manual_seed(7)
    generator = LykenoxVocoderGeneratorV41().eval()
    discriminator = LykenoxMultiScaleWaveformDiscriminator(scales=2).eval()
    mel = torch.randn(1, 16, generator.config.mel_bins)
    f0 = torch.full((1, 16), 100.0)
    voiced = torch.ones(1, 16)
    with torch.no_grad():
        expected = generator(mel, f0, voiced)

    path = tmp_path / "checkpoint.pt"
    save_source_filter_checkpoint(
        path,
        generator,
        discriminator,
        epoch=2,
        global_step=17,
        next_item_offset=3,
        validation_reconstruction_loss=1.0,
        validation_spectral_balance_loss=0.2,
        validation_selection_score=1.1,
        training_provenance=_provenance(),
    )
    restored, _, payload = load_source_filter_checkpoint(path)
    with torch.no_grad():
        actual = restored(mel, f0, voiced)

    assert payload["generator_architecture"] == VOCODER_GENERATOR_V4_1_ARCHITECTURE
    assert payload["epoch"] == 2
    assert payload["global_step"] == 17
    assert payload["next_item_offset"] == 3
    assert torch.equal(expected, actual)


def test_v41_checkpoint_rejects_stale_pitch_contract(tmp_path: Path) -> None:
    generator = LykenoxVocoderGeneratorV41()
    discriminator = LykenoxMultiScaleWaveformDiscriminator(scales=2)
    provenance = _provenance()
    provenance["pitch_target_version"] = "stale"
    path = tmp_path / "bad.pt"

    try:
        save_source_filter_checkpoint(
            path,
            generator,
            discriminator,
            epoch=0,
            global_step=0,
            next_item_offset=0,
            validation_reconstruction_loss=None,
            validation_spectral_balance_loss=None,
            validation_selection_score=None,
            training_provenance=provenance,
        )
    except RuntimeError as exc:
        assert "pitch_target_version" in str(exc)
    else:
        raise AssertionError("stale pitch contract should be rejected")
