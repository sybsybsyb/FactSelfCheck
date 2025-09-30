import numpy as np
import torch
from tokenizers import Tokenizer


class AttnScorePredictor:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def predict_sentences(
        self, sentences: list[str], input_ids: torch.Tensor, attn_diag: torch.Tensor
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        sentences_ranges = self._get_sentences_ranges(input_ids, sentences)
        attn_score_absolute = []
        attn_score_relative = []
        for s_range in sentences_ranges:
            attn_score_absolute.append(self._calc_attn_score(attn_diag[:, :, : s_range[1]]))
            attn_score_relative.append(
                self._calc_attn_score(attn_diag[:, :, s_range[0] : s_range[1]])
            )
        return attn_score_absolute, attn_score_relative

    def _calc_attn_score(self, attn_diag: torch.Tensor) -> np.ndarray:
        return attn_diag.float().log().mean(dim=-1).sum(dim=-1).numpy()

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
