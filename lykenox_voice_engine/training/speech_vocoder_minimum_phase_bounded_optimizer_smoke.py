"""Two-update real-data optimizer smoke for the owned minimum-phase vocoder path.

This is not persistent training.  It is the only optimizer-bearing gate authorized after the
predictor structural smoke: one deterministic owned V2 train segment, exactly two in-memory
SGD updates, no checkpoint load/save, no trainer, and no product-quality claim.

The smoke proves that the instantiated frame-rate predictor can move the already-proven fixed
renderer toward a paired owned waveform under the frozen four-objective Loss V2 contract
without NaN/Inf, output-length regression, severe frame-grid excess, or checkpoint mutation.
Tiny-smoke loss decrease is trainability evidence only and can never accept voice quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from lykenox_voice_engine.models.speech import LykenoxSpeechConfig
from lykenox_voice_engine.models.vocoder.network_minimum_phase_v1 import (
    PREDICTOR_ARCHITECTURE,
    LykenoxFrameRateCepstralPredictorV1,
)
from lykenox_voice_engine.training import speech_vocoder_minimum_phase_renderer as renderer
from lykenox_voice_engine.training.speech_vocoder_data import (
    OWNED_VOCODER_SEGMENT_CONTRACT_VERSION,
    collect_owned_vocoder_segments,
)
from lykenox_voice_engine.training.speech_vocoder_grid_artifact import (
    frame_grid_artifact_excess_metrics,
)
from lykenox_voice_engine.training.speech_vocoder_loss_v2 import (
    ConditioningAlignedLogMelEnvelopeLossV2,
    OWNED_VOCODER_LOSS_V2_VERSION,
    valid_context_multi_resolution_reconstruction_loss,
)
from lykenox_voice_engine.training.speech_vocoder_loss_v2_weight_contract import (
    FROZEN_WEIGHTS,
    OWNED_VOCODER_LOSS_V2_WEIGHT_CONTRACT_VERSION,
    combine_owned_vocoder_loss_v2,
)
from lykenox_voice_engine.training.speech_vocoder_presence_v2 import (
    OWNED_VOCODER_PRESENCE_V2_VERSION,
    target_relative_presence_loss_v2,
)
from lykenox_voice_engine.training.speech_vocoder_source_balance import (
    target_relative_spectral_balance_loss,
)


SMOKE_VERSION = "owned-minimum-phase-bounded-optimizer-smoke-v1"
ARCHITECTURE_CONTRACT_VERSION = "owned-vocoder-architecture-contract-v1"
RENDERER_VERSION = "owned-minimum-phase-time-varying-renderer-v1"
SEGMENT_MEL_FRAMES = 32
MAX_ITEMS = 1
MAX_UPDATES = 2
LEARNING_RATE = 2.0e-4
MAX_GRAD_NORM = 1.0
DATA_SEED = 20260901
MODEL_SEED = 20260903
NOISE_SEED = 97
BOUNDED_OPTIMIZER_SMOKE_AUTHORIZED = True
TRAINER_IMPLEMENTATION_AUTHORIZED = False
PERSISTENT_TRAINING_AUTHORIZED = False
NEW_VOCODER_CHECKPOINT_AUTHORIZED = False


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _protected(root: Path) -> dict[str, Path]:
    training = root / "models" / "lykenox_identity" / "training"
    return {
        "v4_2_best": training / "vocoder_source_filter_v4_2" / "best.pt",
        "v6_best": training / "vocoder_direct_waveform_v6" / "best.pt",
        "v7_best": training / "vocoder_source_free_v7_first_epoch" / "best.pt",
        "acoustic_v2_best": training / "acoustic_frame_context_v2" / "best.pt",
    }


def _parameter_vector(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([parameter.detach().reshape(-1).cpu() for parameter in model.parameters()])


def _finite_scalar(value: torch.Tensor) -> bool:
    return value.ndim == 0 and bool(torch.isfinite(value))


def _forward_objectives(
    model: LykenoxFrameRateCepstralPredictorV1,
    *,
    mel: torch.Tensor,
    f0_hz: torch.Tensor,
    voiced: torch.Tensor,
    periodicity: torch.Tensor,
    target: torch.Tensor,
    envelope_objective: ConditioningAlignedLogMelEnvelopeLossV2,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    cepstrum = model(mel, f0_hz, voiced, periodicity)
    prediction, excitation = renderer.render_owned_minimum_phase_vocoder_path(
        cepstrum,
        f0_hz,
        voiced,
        periodicity,
        noise_seed=NOISE_SEED,
    )
    reconstruction = valid_context_multi_resolution_reconstruction_loss(
        prediction,
        target,
    ).total
    envelope = envelope_objective(prediction, mel).total
    presence = target_relative_presence_loss_v2(
        prediction,
        target,
        sample_rate=renderer.SAMPLE_RATE,
    ).loss
    spectral_balance = target_relative_spectral_balance_loss(
        prediction,
        target,
        sample_rate=renderer.SAMPLE_RATE,
    ).loss
    terms = {
        "reconstruction": reconstruction,
        "envelope": envelope,
        "presence": presence,
        "spectral_balance": spectral_balance,
    }
    total = combine_owned_vocoder_loss_v2(**terms)
    return total, terms, prediction, excitation


def _public_terms(terms: dict[str, torch.Tensor]) -> dict[str, float]:
    return {name: round(float(value.detach()), 10) for name, value in terms.items()}


def run_smoke(root: Path) -> dict[str, object]:
    root = Path(root).resolve()
    protected = _protected(root)
    checkpoints_before = {name: _sha256(path) for name, path in protected.items()}

    segments, skipped = collect_owned_vocoder_segments(
        root,
        "train",
        segment_mel_frames=SEGMENT_MEL_FRAMES,
        max_items=MAX_ITEMS,
        seed=DATA_SEED,
    )
    if len(segments) != 1:
        raise RuntimeError("bounded optimizer smoke requires exactly one owned segment")
    segment = segments[0]
    if segment.conditioning_contract_version != OWNED_VOCODER_SEGMENT_CONTRACT_VERSION:
        raise RuntimeError("bounded optimizer smoke received the wrong data contract")

    mel = segment.mel.unsqueeze(0).cpu()
    f0_hz = segment.f0_hz.unsqueeze(0).cpu()
    voiced = segment.voiced.unsqueeze(0).cpu()
    periodicity = segment.periodicity.unsqueeze(0).cpu()
    target = segment.waveform.unsqueeze(0).cpu()

    torch.manual_seed(MODEL_SEED)
    model = LykenoxFrameRateCepstralPredictorV1().cpu()
    envelope_objective = ConditioningAlignedLogMelEnvelopeLossV2(
        LykenoxSpeechConfig()
    ).cpu()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    initial_parameters = _parameter_vector(model)
    with torch.no_grad():
        initial_total, initial_terms, initial_prediction, initial_excitation = _forward_objectives(
            model,
            mel=mel,
            f0_hz=f0_hz,
            voiced=voiced,
            periodicity=periodicity,
            target=target,
            envelope_objective=envelope_objective,
        )
    initial_grid = frame_grid_artifact_excess_metrics(
        initial_prediction,
        initial_excitation,
        sample_rate=renderer.SAMPLE_RATE,
        hop_length=renderer.HOP_LENGTH,
    )

    updates: list[dict[str, object]] = []
    gradients_all_finite = True
    gradient_nonzero_each_update = True
    for update_index in range(MAX_UPDATES):
        optimizer.zero_grad(set_to_none=True)
        total, terms, prediction, excitation = _forward_objectives(
            model,
            mel=mel,
            f0_hz=f0_hz,
            voiced=voiced,
            periodicity=periodicity,
            target=target,
            envelope_objective=envelope_objective,
        )
        if not _finite_scalar(total) or not all(_finite_scalar(value) for value in terms.values()):
            raise RuntimeError("bounded optimizer smoke produced a non-finite loss")
        total.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
        all_present = all(gradient is not None for gradient in gradients)
        all_finite = all(
            gradient is not None and bool(torch.isfinite(gradient).all())
            for gradient in gradients
        )
        raw_grad_norm = float(
            torch.sqrt(
                sum(
                    gradient.detach().square().sum()
                    for gradient in gradients
                    if gradient is not None
                )
            )
        )
        nonzero = math.isfinite(raw_grad_norm) and raw_grad_norm > 0.0
        gradients_all_finite = gradients_all_finite and all_present and all_finite
        gradient_nonzero_each_update = gradient_nonzero_each_update and nonzero
        clipped_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
        )
        optimizer.step()
        updates.append(
            {
                "update_index": update_index + 1,
                "total_before_step": round(float(total.detach()), 10),
                "terms_before_step": _public_terms(terms),
                "raw_gradient_norm": round(raw_grad_norm, 10),
                "clip_returned_gradient_norm": round(clipped_norm, 10),
                "gradients_all_present": all_present,
                "gradients_all_finite": all_finite,
                "gradient_nonzero": nonzero,
                "prediction_samples": int(prediction.shape[-1]),
                "expected_samples": int(SEGMENT_MEL_FRAMES * renderer.HOP_LENGTH),
                "exact_output_length": int(prediction.shape[-1])
                == int(SEGMENT_MEL_FRAMES * renderer.HOP_LENGTH),
                "parameter_update_executed": True,
            }
        )

    with torch.no_grad():
        final_total, final_terms, final_prediction, final_excitation = _forward_objectives(
            model,
            mel=mel,
            f0_hz=f0_hz,
            voiced=voiced,
            periodicity=periodicity,
            target=target,
            envelope_objective=envelope_objective,
        )
    final_parameters = _parameter_vector(model)
    parameter_delta = final_parameters - initial_parameters
    parameter_delta_norm = float(torch.linalg.vector_norm(parameter_delta))
    parameter_delta_max_abs = float(parameter_delta.abs().max())
    final_grid = frame_grid_artifact_excess_metrics(
        final_prediction,
        final_excitation,
        sample_rate=renderer.SAMPLE_RATE,
        hop_length=renderer.HOP_LENGTH,
    )

    checkpoints_after = {name: _sha256(path) for name, path in protected.items()}
    checkpoints_unchanged = checkpoints_before == checkpoints_after
    initial_total_value = float(initial_total)
    final_total_value = float(final_total)
    relative_total_change = (
        (final_total_value - initial_total_value) / max(abs(initial_total_value), 1e-12)
    )

    gates = {
        "owned_v2_data_contract_exact": segment.conditioning_contract_version
        == OWNED_VOCODER_SEGMENT_CONTRACT_VERSION,
        "frozen_loss_contracts_exact": (
            OWNED_VOCODER_LOSS_V2_VERSION
            == "owned-vocoder-loss-v2-valid-context-conditioning-aligned"
            and OWNED_VOCODER_PRESENCE_V2_VERSION
            == "owned-vocoder-presence-v2-valid-context-target-relative"
            and OWNED_VOCODER_LOSS_V2_WEIGHT_CONTRACT_VERSION
            == "owned-vocoder-loss-v2-weight-contract-v1"
        ),
        "exact_update_budget": len(updates) == MAX_UPDATES,
        "all_gradients_present_finite": gradients_all_finite,
        "gradient_nonzero_each_update": gradient_nonzero_each_update,
        "parameter_delta_finite_nonzero": math.isfinite(parameter_delta_norm)
        and parameter_delta_norm > 0.0,
        "total_loss_decreased": final_total_value < initial_total_value,
        "exact_output_length_before_after": (
            int(initial_prediction.shape[-1]) == SEGMENT_MEL_FRAMES * renderer.HOP_LENGTH
            and int(final_prediction.shape[-1]) == SEGMENT_MEL_FRAMES * renderer.HOP_LENGTH
            and all(bool(update["exact_output_length"]) for update in updates)
        ),
        "no_severe_grid_excess_before": not bool(initial_grid.severe_grid_excess.any()),
        "no_severe_grid_excess_after": not bool(final_grid.severe_grid_excess.any()),
        "checkpoints_unchanged": checkpoints_unchanged,
    }
    status = "pass" if all(gates.values()) else "fail"
    return {
        "status": status,
        "smoke_version": SMOKE_VERSION,
        "architecture_contract_version": ARCHITECTURE_CONTRACT_VERSION,
        "renderer_version": RENDERER_VERSION,
        "predictor_architecture": PREDICTOR_ARCHITECTURE,
        "data_contract_version": OWNED_VOCODER_SEGMENT_CONTRACT_VERSION,
        "loss_contract_version": OWNED_VOCODER_LOSS_V2_VERSION,
        "presence_contract_version": OWNED_VOCODER_PRESENCE_V2_VERSION,
        "loss_weight_contract_version": OWNED_VOCODER_LOSS_V2_WEIGHT_CONTRACT_VERSION,
        "frozen_weights": FROZEN_WEIGHTS.as_dict(),
        "bounded_contract": {
            "split": "train",
            "segment_mel_frames": SEGMENT_MEL_FRAMES,
            "max_items": MAX_ITEMS,
            "max_updates": MAX_UPDATES,
            "learning_rate": LEARNING_RATE,
            "max_gradient_norm": MAX_GRAD_NORM,
            "data_seed": DATA_SEED,
            "model_seed": MODEL_SEED,
            "noise_seed": NOISE_SEED,
        },
        "segment": {
            "utterance_id": segment.utterance_id,
            "start_frame": segment.start_frame,
            "mel_frames": segment.mel_frames,
            "skipped_item_count": len(skipped),
        },
        "initial": {
            "total": round(initial_total_value, 10),
            "terms": _public_terms(initial_terms),
            "hop_autocorrelation_excess": float(initial_grid.hop_autocorrelation_excess.max()),
            "double_hop_autocorrelation_excess": float(
                initial_grid.double_hop_autocorrelation_excess.max()
            ),
            "grid_harmonic_power_fraction_excess": float(
                initial_grid.grid_harmonic_power_fraction_excess.max()
            ),
            "severe_grid_excess": bool(initial_grid.severe_grid_excess.any()),
        },
        "updates": updates,
        "final": {
            "total": round(final_total_value, 10),
            "terms": _public_terms(final_terms),
            "relative_total_change": round(relative_total_change, 10),
            "parameter_delta_norm": round(parameter_delta_norm, 10),
            "parameter_delta_max_abs": round(parameter_delta_max_abs, 10),
            "hop_autocorrelation_excess": float(final_grid.hop_autocorrelation_excess.max()),
            "double_hop_autocorrelation_excess": float(
                final_grid.double_hop_autocorrelation_excess.max()
            ),
            "grid_harmonic_power_fraction_excess": float(
                final_grid.grid_harmonic_power_fraction_excess.max()
            ),
            "severe_grid_excess": bool(final_grid.severe_grid_excess.any()),
        },
        "gates": gates,
        "checkpoints_unchanged": checkpoints_unchanged,
        "model_instantiated": True,
        "optimizer_created": True,
        "bounded_optimizer_smoke_started": True,
        "parameter_update_executed": True,
        "update_count": len(updates),
        "trainer_instantiated": False,
        "checkpoint_loaded": False,
        "checkpoint_saved": False,
        "persistent_training_started": False,
        "persistent_training_authorized": False,
        "new_vocoder_checkpoint_authorized": False,
        "metrics_accept_voice_quality": False,
        "third_party_model_used": False,
        "predicted_duration_modified": False,
        "posthoc_gain_normalization_used": False,
        "posthoc_eq_used": False,
        "posthoc_denoising_used": False,
        "next_gate": (
            "review_bounded_optimizer_smoke_before_authorizing_any_trainability_gate"
            if status == "pass"
            else "revise_predictor_or_optimizer_smoke_before_any_training_authorization"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(run_smoke(args.root), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
