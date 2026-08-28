"""Bounded inference gate for LYKENOX predicted-duration semantics.

This gate runs after persistent acoustic frame-context v2 has passed its teacher-duration
held-out audit. It does not train anything. It proves that product-side duration
regulation no longer uses the historical fixed 1..80 clamp and that the accepted v2
checkpoint can run from text tokens alone with predicted mel/F0/voicing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from lykenox_voice_engine.core.spanish_g2p import TOKEN_TO_ID
from lykenox_voice_engine.core.spanish_text_frontend import SpanishTextFrontend
from lykenox_voice_engine.models.speech.config import FRAME_CONTEXT_TOKEN_PROGRESS_CONV_V1
from lykenox_voice_engine.models.speech.duration_policy import (
    CONTENT_MAX_DURATION_FRAMES,
    PAUSE_MAX_DURATION_FRAMES,
    PREDICTED_DURATION_POLICY_VERSION,
    STRUCTURAL_MAX_DURATION_FRAMES,
    regulate_predicted_durations,
)
from lykenox_voice_engine.training.speech_acoustic_prosody_artifact import (
    load_acoustic_prosody_checkpoint,
)
from lykenox_voice_engine.training.speech_acoustic_frame_context_train import (
    TRAINER_CONTRACT_VERSION,
)


SMOKE_VERSION = "predicted-duration-semantics-smoke-v1"


def _policy_probe() -> dict[str, object]:
    token_ids = torch.tensor(
        [[
            TOKEN_TO_ID["<pad>"],
            TOKEN_TO_ID["<bos>"],
            TOKEN_TO_ID["<eos>"],
            TOKEN_TO_ID["<wb>"],
            TOKEN_TO_ID["<pau_short>"],
            TOKEN_TO_ID["<pau_long>"],
            TOKEN_TO_ID["a"],
            TOKEN_TO_ID["e"],
        ]],
        dtype=torch.long,
    )
    token_mask = torch.tensor(
        [[False, True, True, True, True, True, True, True]],
        dtype=torch.bool,
    )
    raw = torch.tensor(
        [[200.0, 0.10, 1.60, 0.49, 0.10, 500.0, 0.10, 120.0]],
        dtype=torch.float32,
    )
    regulated = regulate_predicted_durations(token_ids, token_mask, raw)
    expected = torch.tensor(
        [[0, 0, 2, 0, 1, PAUSE_MAX_DURATION_FRAMES, 1, 120]],
        dtype=torch.long,
    )
    return {
        "exact_expected": bool(torch.equal(regulated, expected)),
        "regulated": [int(value) for value in regulated[0].tolist()],
        "padding_forced_zero": int(regulated[0, 0]) == 0,
        "bos_zero_supported": int(regulated[0, 1]) == 0,
        "structural_positive_supported": int(regulated[0, 2]) == 2,
        "wb_zero_supported": int(regulated[0, 3]) == 0,
        "pause_min_one": int(regulated[0, 4]) == 1,
        "pause_safety_cap": int(regulated[0, 5]) == PAUSE_MAX_DURATION_FRAMES,
        "content_min_one": int(regulated[0, 6]) == 1,
        "content_above_legacy_80_preserved": int(regulated[0, 7]) == 120,
    }


def run_predicted_duration_semantics_smoke(root: Path) -> dict[str, object]:
    root = Path(root).resolve()
    checkpoint = (
        root
        / "models"
        / "lykenox_identity"
        / "training"
        / "acoustic_frame_context_v2"
        / "best.pt"
    )
    if not checkpoint.exists():
        raise FileNotFoundError(f"Persistent acoustic v2 best checkpoint not found: {checkpoint}")

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    model, payload = load_acoustic_prosody_checkpoint(checkpoint)
    run_config = payload.get("run_config")
    if not isinstance(run_config, dict):
        raise RuntimeError("v2 checkpoint is missing run_config")
    architecture_exact = (
        model.config.frame_context_version == FRAME_CONTEXT_TOKEN_PROGRESS_CONV_V1
        and run_config.get("trainer_contract_version") == TRAINER_CONTRACT_VERSION
        and run_config.get("frame_context_version") == FRAME_CONTEXT_TOKEN_PROGRESS_CONV_V1
    )
    if not architecture_exact:
        raise RuntimeError("Predicted-duration smoke requires the accepted persistent v2 checkpoint")

    policy = _policy_probe()

    # Teacher timing remains a separate exact training/audit contract and must never be
    # routed through the inference policy. Include zero-duration structural tokens and
    # values above the historical 80-frame ceiling to prove exact preservation.
    teacher_token_ids = torch.tensor(
        [[
            TOKEN_TO_ID["<bos>"],
            TOKEN_TO_ID["a"],
            TOKEN_TO_ID["<wb>"],
            TOKEN_TO_ID["e"],
            TOKEN_TO_ID["<pau_long>"],
            TOKEN_TO_ID["<eos>"],
        ]],
        dtype=torch.long,
    )
    teacher_mask = torch.ones_like(teacher_token_ids, dtype=torch.bool)
    teacher_durations = torch.tensor([[0, 99, 0, 120, 7, 2]], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        teacher_output = model(teacher_token_ids, teacher_mask, teacher_durations)
    teacher_exact = torch.equal(
        teacher_output["regulated_durations"],
        teacher_durations,
    )
    teacher_length_exact = int(teacher_output["mel_lengths"][0]) == int(teacher_durations.sum())

    frontend = SpanishTextFrontend()
    probe_texts = [
        "La voz de Lykenox debe conservar un ritmo natural y estable.",
        "Esta prueba usa únicamente texto para predecir duración, tono y sonoridad.",
        "Hola mundo, hoy comprobamos pausas, palabras y una frase un poco más larga.",
    ]
    text_runs: list[dict[str, object]] = []
    inference_finite = True
    inference_lengths_exact = True
    for text in probe_texts:
        processed = frontend.process(text)
        token_ids = torch.tensor([processed.token_ids], dtype=torch.long)
        token_mask = torch.ones_like(token_ids, dtype=torch.bool)
        with torch.no_grad():
            output = model(token_ids, token_mask)
        regulated = output["regulated_durations"]
        length = int(output["mel_lengths"][0])
        exact_length = length == int(regulated.sum())
        finite = bool(
            torch.isfinite(output["mel"]).all()
            and torch.isfinite(output["f0_prediction_hz"]).all()
            and torch.isfinite(output["voicing_logits"]).all()
            and torch.isfinite(output["duration_prediction"]).all()
        )
        inference_finite = inference_finite and finite
        inference_lengths_exact = inference_lengths_exact and exact_length
        valid_durations = regulated[0, token_mask[0]]
        text_runs.append(
            {
                "text": text,
                "token_count": len(processed.token_ids),
                "mel_frames": length,
                "regulated_duration_sum": int(regulated.sum()),
                "min_token_duration": int(valid_durations.min()),
                "max_token_duration": int(valid_durations.max()),
                "zero_duration_token_count": int((valid_durations == 0).sum()),
                "finite_outputs": finite,
                "length_contract_exact": exact_length,
            }
        )

    checks = {
        "architecture_identity_exact": architecture_exact,
        "policy_probe_exact": bool(policy["exact_expected"]),
        "padding_forced_zero": bool(policy["padding_forced_zero"]),
        "structural_zero_duration_supported": bool(
            policy["bos_zero_supported"] and policy["wb_zero_supported"]
        ),
        "structural_positive_duration_supported": bool(policy["structural_positive_supported"]),
        "content_min_one": bool(policy["content_min_one"]),
        "pause_min_one": bool(policy["pause_min_one"]),
        "content_above_legacy_80_preserved": bool(policy["content_above_legacy_80_preserved"]),
        "teacher_durations_preserved_exactly": bool(teacher_exact and teacher_length_exact),
        "text_only_inference_outputs_finite": inference_finite,
        "predicted_duration_sum_matches_mel_length": inference_lengths_exact,
    }
    status = "pass" if all(checks.values()) else "needs_review"
    return {
        "status": status,
        "device": "cpu",
        "smoke_version": SMOKE_VERSION,
        "predicted_duration_policy_version": PREDICTED_DURATION_POLICY_VERSION,
        "checkpoint": str(checkpoint),
        "trainer_contract_version": TRAINER_CONTRACT_VERSION,
        "frame_context_version": model.config.frame_context_version,
        "legacy_checkpoint_max_duration_frames": int(model.config.max_duration_frames),
        "content_max_duration_frames": CONTENT_MAX_DURATION_FRAMES,
        "structural_max_duration_frames": STRUCTURAL_MAX_DURATION_FRAMES,
        "pause_max_duration_frames": PAUSE_MAX_DURATION_FRAMES,
        **checks,
        "policy_probe": policy,
        "teacher_duration_probe": {
            "durations": [int(value) for value in teacher_durations[0].tolist()],
            "regulated": [int(value) for value in teacher_output["regulated_durations"][0].tolist()],
            "mel_frames": int(teacher_output["mel_lengths"][0]),
        },
        "text_only_runtime_probes": text_runs,
        "reference_audio_required": False,
        "waveform_pitch_target_required": False,
        "next_gate": (
            "build_reference_free_text_to_waveform_smoke"
            if status == "pass"
            else "fix_predicted_duration_policy_before_end_to_end"
        ),
        "warning": (
            "A pass validates inference duration semantics and text-only acoustic execution. "
            "It does not yet validate waveform quality; the next gate must connect predicted "
            "mel/F0/voicing to the accepted persistent v4.1 vocoder."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(
        json.dumps(
            run_predicted_duration_semantics_smoke(args.root),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
