from typing import Callable

import torch
from tokenizers import Tokenizer


class ProbaScorePredictor:
    def __init__(self, tokenizer: Tokenizer, agg: Callable[[torch.Tensor | list[float]], float]):
        self.tokenizer = tokenizer
        self.agg = agg

    def predict_sentences(
        self,
        sentences: list[str],
        input_ids: torch.Tensor,
        proba_metrics: torch.Tensor | list[float],
    ) -> list[float]:
        assert len(input_ids) - 1 == len(proba_metrics)
        sentences_ranges = self._get_sentences_ranges(input_ids, sentences)
        proba_scores = []
        for s_range in sentences_ranges:
            proba_scores.append(
                self._calc_proba_score(proba_metrics[s_range[0] - 1 : s_range[1] - 1])
            )
        return proba_scores

    def _calc_proba_score(self, proba_metrics: torch.Tensor | list[float]) -> float:
        return self.agg(proba_metrics)

    def _get_sentences_ranges(
        self, input_ids: torch.Tensor, sentences: list[str]
    ) -> list[tuple[int, int]]:
        sentences_ranges = []

        current_end = 0
        for i, s in enumerate(sentences):
            if i > 0:
                s = " " + s
            [sentence_ids] = self.tokenizer.encode(s, add_special_tokens=False, return_tensors="pt")

            current_input_ids = input_ids[current_end:]
            start, end = self._find_subtensor(sentence_ids, current_input_ids)
            start = start + current_end
            end = end + current_end

            sentences_ranges.append((start, end))
            current_end = end

        for i, s_range in enumerate(sentences_ranges):
            if s_range[0] is None or s_range[0] < 0:
                raise ValueError(f"Sentence {i} has invalid range {s_range}")

        for i in range(len(sentences_ranges) - 1):
            assert sentences_ranges[i][0] < sentences_ranges[i + 1][0]
            assert sentences_ranges[i][1] <= sentences_ranges[i + 1][0]

        return sentences_ranges

    def _find_subtensor(self, pattern: torch.Tensor, vector: torch.Tensor) -> tuple[int, int]:
        if len(pattern) == 0:
            raise ValueError("Pattern is empty.")
        pattern_len = len(pattern)
        for i in range(len(vector) - pattern_len + 1):
            if torch.all(vector[i : i + pattern_len] == pattern):
                return i, i + pattern_len
        raise ValueError("Pattern not found in vector.")
