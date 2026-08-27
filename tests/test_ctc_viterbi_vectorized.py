from __future__ import annotations

import math
import unittest

import torch
import torch.nn.functional as F

from lykenox_voice_engine.core.ctc_alignment import ctc_viterbi_state_path


def _reference_viterbi(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    blank_id: int,
) -> tuple[torch.Tensor, float]:
    probabilities = log_probs.detach().cpu()
    target_values = [int(value) for value in targets.detach().cpu().tolist()]
    expanded: list[int] = [blank_id]
    for token in target_values:
        expanded.extend((token, blank_id))

    time_steps = int(probabilities.shape[0])
    state_count = len(expanded)
    previous = [float("-inf")] * state_count
    previous[0] = float(probabilities[0, blank_id].item())
    previous[1] = float(probabilities[0, target_values[0]].item())
    backpointer: list[list[int]] = [[0] * state_count for _ in range(time_steps)]

    for time_index in range(1, time_steps):
        current = [float("-inf")] * state_count
        for state in range(state_count):
            candidates: list[tuple[float, int]] = [(previous[state], 0)]
            if state > 0:
                candidates.append((previous[state - 1], 1))
            if (
                state > 1
                and expanded[state] != blank_id
                and expanded[state] != expanded[state - 2]
            ):
                candidates.append((previous[state - 2], 2))
            best_score, move = max(candidates, key=lambda item: item[0])
            current[state] = best_score + float(
                probabilities[time_index, expanded[state]].item()
            )
            backpointer[time_index][state] = move
        previous = current

    final_states = (state_count - 1, state_count - 2)
    state = max(final_states, key=lambda index: previous[index])
    score = previous[state]
    if not math.isfinite(score):
        raise RuntimeError("reference path is not finite")

    path = torch.empty((time_steps,), dtype=torch.long)
    path[-1] = state
    for time_index in range(time_steps - 1, 0, -1):
        state -= backpointer[time_index][state]
        path[time_index - 1] = state
    return path, score


class VectorizedViterbiTests(unittest.TestCase):
    def test_matches_previous_reference_algorithm(self) -> None:
        torch.manual_seed(1234)
        blank_id = 8
        cases = [
            torch.tensor([1, 2, 3], dtype=torch.long),
            torch.tensor([2, 2, 4], dtype=torch.long),
            torch.tensor([5], dtype=torch.long),
            torch.tensor([1, 3, 3, 2], dtype=torch.long),
        ]
        for targets in cases:
            time_steps = 18
            logits = torch.randn(time_steps, blank_id + 1)
            log_probs = F.log_softmax(logits, dim=-1)
            expected_path, expected_score = _reference_viterbi(
                log_probs,
                targets,
                blank_id,
            )
            actual_path, actual_score = ctc_viterbi_state_path(
                log_probs,
                targets,
                blank_id,
            )
            self.assertTrue(torch.equal(actual_path, expected_path))
            self.assertAlmostEqual(actual_score, expected_score, places=5)


if __name__ == "__main__":
    unittest.main()
