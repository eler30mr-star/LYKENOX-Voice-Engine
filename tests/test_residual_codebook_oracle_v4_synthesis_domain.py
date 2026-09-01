from __future__ import annotations

import inspect

import torch

from scripts.diagnostic_residual_codebook_oracle_v4_synthesis_domain import (
    DIAGNOSTIC_VERSION,
    LOCAL_CONV_FFT_SIZE,
    _local_filtered_vector_response,
)
from lykenox_voice_engine.training.speech_residual_codebook_v1 import (
    CODEVECTOR_SAMPLES,
    POLICY_ID,
    _sqrt_hann,
    residual_synthesis_from_analysis_vectors,
)
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_renderer import (
    CEPSTRAL_ORDER,
    HOP_LENGTH,
    N_FFT,
    one_sided_real_cepstrum_to_minimum_phase_fir,
    render_time_varying_minimum_phase,
)


def _affected_block_range(vector_index: int, frame_count: int) -> tuple[int, int]:
    total_samples = frame_count * HOP_LENGTH
    original_start = (vector_index - 1) * HOP_LENGTH
    left_clip = max(0, -original_start)
    right_clip = max(0, original_start + CODEVECTOR_SAMPLES - total_samples)
    retained = CODEVECTOR_SAMPLES - left_clip - right_clip
    global_start = max(0, original_start)
    convolution_length = retained + N_FFT - 1
    support_end = min(total_samples, global_start + convolution_length)
    return global_start // HOP_LENGTH, (support_end - 1) // HOP_LENGTH


def test_local_filtered_response_matches_full_renderer_single_vector_contribution() -> None:
    torch.manual_seed(7)
    frame_count = 12
    cepstrum = torch.randn(frame_count, CEPSTRAL_ORDER, dtype=torch.float32) * 0.01
    filters = one_sided_real_cepstrum_to_minimum_phase_fir(cepstrum, n_fft=N_FFT)
    filter_fft = torch.fft.rfft(filters, n=LOCAL_CONV_FFT_SIZE, dim=-1)

    for vector_index in (0, 1, 5, frame_count):
        vector = torch.randn(1, CODEVECTOR_SAMPLES, dtype=torch.float32)
        local = _local_filtered_vector_response(
            vector,
            vector_index=vector_index,
            filter_fft=filter_fft,
            frame_count=frame_count,
        )

        all_vectors = torch.zeros(
            frame_count + 1,
            CODEVECTOR_SAMPLES,
            dtype=torch.float32,
        )
        all_vectors[vector_index] = vector[0]
        excitation = residual_synthesis_from_analysis_vectors(
            all_vectors,
            output_samples=frame_count * HOP_LENGTH,
        )
        full = render_time_varying_minimum_phase(
            excitation.unsqueeze(0),
            cepstrum.unsqueeze(0),
            hop_length=HOP_LENGTH,
            n_fft=N_FFT,
        ).squeeze(0)
        first_block, last_block = _affected_block_range(vector_index, frame_count)
        expected = full[first_block * HOP_LENGTH : (last_block + 1) * HOP_LENGTH]
        assert local.shape == (1, expected.numel())
        assert torch.allclose(local[0], expected, atol=2.0e-5, rtol=2.0e-5)


def test_synthesis_domain_oracle_policy_and_no_training_contract() -> None:
    import scripts.diagnostic_residual_codebook_oracle_v4_synthesis_domain as module

    source = inspect.getsource(module)
    assert DIAGNOSTIC_VERSION == "owned-residual-codebook-heldout-oracle-v4-synthesis-domain"
    assert POLICY_ID == "LYX-POL-001"
    assert "training_executed\": False" in source
    assert "optimizer_created\": False" in source
    assert "checkpoint_written\": False" in source
    assert "posthoc_output_gain_normalization_used\": False" in source
    assert "oracle_indices_or_gains_valid_for_product_inference\": False" in source
    assert "MAX_ORACLE_GAIN" not in source
    assert "from_pretrained" not in source
    assert "remote_inference_used\": False" in source


def test_sqrt_hann_codevector_geometry_remains_512_over_256() -> None:
    window = _sqrt_hann(dtype=torch.float32)
    assert int(window.numel()) == CODEVECTOR_SAMPLES == 2 * HOP_LENGTH
    assert HOP_LENGTH == 256
