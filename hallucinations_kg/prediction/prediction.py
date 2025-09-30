import json
from typing import Any, Callable

from hallucinations_kg.models.baselines.selfcheckgpt import (
    SelfCheckGPTNLIPredictor,
    SelfCheckGPTPredictor,
)
from hallucinations_kg.models.predictor import (
    FactPredictor,
    Graph,
)
from hallucinations_kg.models.prompt_predictor import FactPromptPredictor, FactTextPromptPredictor
from hallucinations_kg.models.sentence_predictor import (
    MostFrequentSentencePredictor,
    SentenceAggregatePredictor,
    SentencePredictor,
)


def add_sentence_predictor_results(
    entry: dict[str, Any],
    predictor_aggregation_functions: list[
        tuple[type[SentencePredictor] | type[FactPredictor], Callable[[list[float]], float] | None]
    ],
    sentences_column: str,
    samples_column: str,
    y_true: list[float] | None = None,
    n_samples: int | None = None,
) -> dict[str, Any]:
    sentences_graphs = [Graph.from_triples(g) for g in entry[f"{sentences_column}_graph"]]
    samples_graphs = [Graph.from_triples(g) for g in entry[f"{samples_column}_graph"]]
    fact_prompt_answers = entry["fact_prompt_answers"]
    fact_text_prompt_answers = entry["fact_text_prompt_answers"]
    selfcheckgpt_answers = entry["selfcheckgpt_answers"]
    selfcheckgpt_nli_scores = entry["selfcheckgpt_nli_scores"]
    for predictor_cls, aggregation_function in predictor_aggregation_functions:
        sentence_predictor = _get_sentence_predictor(
            predictor_cls, aggregation_function, y_true, n_samples
        )
        sentences_answers = _get_sentences_answers(
            predictor_cls,
            fact_prompt_answers,
            fact_text_prompt_answers,
            selfcheckgpt_answers,
            selfcheckgpt_nli_scores,
        )
        predictions = []
        for g, answers in zip(sentences_graphs, sentences_answers):
            prediction = sentence_predictor.predict(
                g, samples_graphs, sentences_graphs=sentences_graphs, answers=answers
            )
            predictions.append(prediction)
        entry[f"sent_score__{json.dumps(sentence_predictor.params())}"] = predictions
    return entry


def _get_sentence_predictor(
    predictor_cls: type[SentencePredictor] | type[FactPredictor],
    aggregation_function: Callable[[list[float]], float] | None,
    y_true: list[float] | None,
    n_samples: int | None,
) -> SentencePredictor:
    sentence_predictor: SentencePredictor
    if issubclass(predictor_cls, FactPredictor) or issubclass(predictor_cls, FactPromptPredictor):
        fact_predictor = predictor_cls()
        assert aggregation_function is not None
        sentence_predictor = SentenceAggregatePredictor(
            fact_predictor, aggregation_function, n_samples=n_samples
        )
    elif predictor_cls == MostFrequentSentencePredictor:
        sentence_predictor = predictor_cls(y_true=y_true)
    elif predictor_cls == SelfCheckGPTPredictor or predictor_cls == SelfCheckGPTNLIPredictor:
        sentence_predictor = predictor_cls(n_samples=n_samples)
    else:
        sentence_predictor = predictor_cls()
    return sentence_predictor


def _get_sentences_answers(
    predictor_cls: type[SentencePredictor] | type[FactPredictor],
    fact_prompt_answers: list[list[str]],
    fact_text_prompt_answers: list[list[str]],
    selfcheckgpt_answers: list[list[str]],
    selfcheckgpt_nli_scores: list[list[float]] | None = None,
) -> list[list[str | float | None]]:
    sentences_answers: list[list[str | float | None]]
    if predictor_cls == FactPromptPredictor:
        sentences_answers = fact_prompt_answers  # type: ignore
    elif predictor_cls == FactTextPromptPredictor:
        sentences_answers = fact_text_prompt_answers  # type: ignore
    elif predictor_cls == SelfCheckGPTPredictor:
        sentences_answers = selfcheckgpt_answers  # type: ignore
    elif predictor_cls == SelfCheckGPTNLIPredictor:
        assert selfcheckgpt_nli_scores is not None
        sentences_answers = selfcheckgpt_nli_scores  # type: ignore
    else:
        # In this case, the sentences answers are not needed
        sentences_answers = [[None] * len(answers) for answers in fact_prompt_answers]
    return sentences_answers


def add_fact_predictor_results(
    entry: dict[str, Any],
    predictor_aggregation_functions: list[type[SentencePredictor] | type[FactPredictor]],
) -> dict[str, Any]:
    sentences_graphs = [Graph.from_triples(g) for g in entry["gpt3_sentences_graph"]]
    samples_graphs = [Graph.from_triples(g) for g in entry["gpt3_text_samples_graph"]]
    fact_prompt_answers = entry["fact_prompt_answers"]
    fact_text_prompt_answers = entry["fact_text_prompt_answers"]
    selfcheckgpt_answers = entry["selfcheckgpt_answers"]
    predictor: FactPredictor | SentencePredictor
    for predictor_cls in predictor_aggregation_functions:
        predictor = predictor_cls()
        sentences_answers = _get_sentences_answers(
            predictor_cls, fact_prompt_answers, fact_text_prompt_answers, selfcheckgpt_answers
        )
        predictions = []
        for sentence_graph, answers in zip(sentences_graphs, sentences_answers):
            sentence_facts_predictions = []
            if predictor_cls != SelfCheckGPTPredictor:
                for fact, fact_answers in zip(sentence_graph.triples, answers):
                    assert isinstance(predictor, FactPredictor)
                    prediction = predictor.predict(fact, samples_graphs, answers=fact_answers)
                    sentence_facts_predictions.append(prediction)
            else:
                assert isinstance(predictor, SelfCheckGPTPredictor)
                sentence_prediction = predictor.predict(
                    sentence_graph, samples_graphs, answers=answers
                )
                sentence_facts_predictions = [sentence_prediction] * len(sentence_graph.triples)
            predictions.append(sentence_facts_predictions)
        entry[f"fact_score__{json.dumps(predictor.params())}"] = predictions
    return entry
