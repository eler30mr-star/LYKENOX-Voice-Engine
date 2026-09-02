from __future__ import annotations

import torch

from scripts.diagnostic_residual_codebook_oracle_v5_synthesis_domain_coherent import (
    CandidateSet,
    _beam_select_sequence,
    _filtered_overlap_discontinuity,
)
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_renderer import HOP_LENGTH


def _candidate_set(*, local_mse: list[float], responses: torch.Tensor) -> CandidateSet:
    count = len(local_mse)
    return CandidateSet(
        indices=torch.arange(count, dtype=torch.long),
        scaled_vectors=torch.zeros(count, HOP_LENGTH * 2, dtype=torch.float32),
        scaled_responses=responses.to(torch.float32).contiguous(),
        local_mse=torch.tensor(local_mse, dtype=torch.float32),
        response_cosine=torch.ones(count, dtype=torch.float32),
        gains=torch.ones(count, dtype=torch.float32),
        global_response_start=0,
    )


def test_filtered_overlap_cost_uses_exactly_one_256_sample_shared_region() -> None:
    previous_response = torch.zeros(1, HOP_LENGTH * 2)
    current_response = torch.ones(1, HOP_LENGTH * 2)
    # Samples outside the shared [0, 256) region must not influence the transition cost.
    current_response[:, HOP_LENGTH:] = 100.0
    previous = _candidate_set(local_mse=[0.0], responses=previous_response)
    current = _candidate_set(local_mse=[0.0], responses=current_response)

    cost = _filtered_overlap_discontinuity(
        previous,
        current,
        current_vector_index=1,
    )

    assert cost.shape == (1, 1)
    assert torch.allclose(cost, torch.ones_like(cost), atol=1.0e-7, rtol=0.0)


def test_beam_search_can_reject_locally_best_but_incoherent_path() -> None:
    zero = torch.zeros(HOP_LENGTH * 2)
    ten = torch.full((HOP_LENGTH * 2,), 10.0)
    first = _candidate_set(
        local_mse=[0.0, 0.2],
        responses=torch.stack((zero, ten), dim=0),
    )
    second = _candidate_set(
        local_mse=[0.0, 0.2],
        responses=torch.stack((ten, zero), dim=0),
    )

    states, transition_costs = _beam_select_sequence(
        [first, second],
        beam_size=4,
        continuity_weight=1.0,
    )

    # Independent greedy selection would be [0, 0], with a very large filtered discontinuity.
    # The coherent global path pays a small local penalty to keep adjacent filtered responses aligned.
    assert states == [0, 1]
    assert transition_costs == [0.0, 0.0]
