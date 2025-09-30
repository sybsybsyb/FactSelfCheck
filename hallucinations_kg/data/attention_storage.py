from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from torch import Tensor
from tqdm import tqdm
from transformers.modeling_outputs import CausalLMOutputWithPast


class ActivationStorage(ABC):
    """Extract intermediate states of an LLM and save them to disk."""

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        return self.update(*args, **kwargs)

    @abstractmethod
    def update(
        self,
        outputs: CausalLMOutputWithPast,
        **kwargs: Any,
    ) -> None:
        raise NotImplementedError()

    def flush(self) -> None:
        pass


class AllActivationsStorage(ActivationStorage):
    """Saves all activations to disk.
    The saved hidden_states has shape: (num_layers, [batch_size, sequence_length, hidden_size])
    """

    def __init__(
        self,
        save_dir: Path,
        max_save_workers: int,
        verbose: bool = True,
    ):
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._check_save_dir()

        self.verbose = verbose

        self.max_save_workers = max_save_workers
        self.save_executor = ThreadPoolExecutor(max_workers=max_save_workers)

    def flush(self) -> None:
        self.save_executor.shutdown(wait=True)
        if self.verbose:
            files = list(self.save_dir.glob("*.pt"))
            size = sum(file.stat().st_size for file in files)
            logger.info(f"Stored total {size * 1e-9:0.1f}GB in {len(files)} files")

    def save(self, intermediate_states: dict[str, Any], batch_idx: int) -> None:
        self.save_executor.submit(
            self._do_save,
            intermediate_states=intermediate_states,
            batch_idx=batch_idx,
        )

    def _do_save(self, intermediate_states: dict[str, Any], batch_idx: int) -> None:
        save_file = self.save_dir / f"batch_{batch_idx}.pt"
        torch.save(intermediate_states, save_file)
        if self.verbose:
            logger.info(
                f"Saved ({save_file.stat().st_size * 1e-9:0.1f}GB) activations to {save_file}"
            )

    def _check_save_dir(self) -> None:
        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True, exist_ok=True)

        legacy_data = list(self.save_dir.glob("batch_*.pt"))
        if len(legacy_data) > 0:
            raise FileExistsError(
                f"Save directory {self.save_dir} already contain data, remove it first."
            )

        other_data = list(self.save_dir.iterdir())
        if len(other_data) > 0:
            logger.warning(
                f"Save directory {self.save_dir} contains {len(other_data)} files that are not .pt files."
            )


class AttentionAndLaplacianDiagsFeatureStorage(AllActivationsStorage):
    """Saves minimal amount of intermediate data, and attention, laplacian diagonals."""

    def __init__(
        self,
        storage_path: Path,
        max_save_workers: int,
        pad_token_id: int,
        verbose: bool = True,
    ):
        super().__init__(storage_path, max_save_workers, verbose)
        self.pad_token_id = pad_token_id
        self.attention_diags: list[Tensor] = []

    def __repr__(self) -> str:
        return f"{type(self).__name__}(max_save_workers={self.max_save_workers}, pad_token_id={self.pad_token_id}, verbose={self.verbose})"

    def update(
        self,
        outputs: CausalLMOutputWithPast,
        **kwargs: Any,
    ) -> None:
        assert outputs.attentions is not None
        self.extract_and_record_features_data(outputs.attentions)

    def extract_and_record_features_data(
        self,
        attentions: tuple[Tensor, ...],
    ) -> None:
        attentions_cpu = _map_attentions_to_cpu(attentions)
        per_example_attn_matrices = extract_per_example_attention_matrices(
            per_layer_batched_data=attentions_cpu,
        )

        for attn_example in tqdm(
            per_example_attn_matrices,
            desc="Computing attention diagonals",
            leave=False,
        ):
            self.attention_diags.append(attention_diagonal(attn_example))

    def flush(self) -> None:
        super().flush()
        attn_diags_file = self.save_dir / "attn_diags.pt"
        torch.save(self.attention_diags, attn_diags_file)

        logger.info(
            f"Saved ({attn_diags_file.stat().st_size * 1e-9:0.1f}GB) "
            f"attention diagonals to {attn_diags_file}"
        )


def _map_attentions_to_cpu(
    attentions: tuple[Tensor, ...],
) -> tuple[tuple[Tensor, ...], ...]:
    return tuple(
        tuple(layer_attn.cpu() for layer_attn in gen_tok_attn) for gen_tok_attn in attentions
    )


def attention_diagonal(item_attn: list[Tensor]) -> Tensor:
    """Computes attention diagonal for single example from dataset.
    Input shape of item_attn is [#layers, [#heads x seq_length x seq_length]]
    Output shape is [#heads x (#layers * seq_length)]
    """
    return torch.stack([torch.diagonal(layer_attn, dim1=1, dim2=2) for layer_attn in item_attn])


def extract_per_example_attention_matrices(
    per_layer_batched_data: tuple[tuple[Tensor, ...], ...],
) -> list[list[Tensor]]:
    num_layers = len(per_layer_batched_data)
    num_examples = len(per_layer_batched_data[0])
    assert num_examples == 1, "Only batch size 1 is supported, as we do not remove padding tokens"
    results: list[list[Tensor]] = []
    for example_idx in range(num_examples):
        results.append([])
        for layer_idx in range(num_layers):
            attn_scores = per_layer_batched_data[layer_idx][example_idx]
            results[-1].append(attn_scores)

            summed_attn = attn_scores.sum(dim=-1)
            assert torch.isclose(
                summed_attn,
                torch.tensor(1.0, dtype=summed_attn.dtype),
                atol=1e-2,  # due to unknown reasons, the margin is quite large
            ).all()
    return results
