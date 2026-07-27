#!/usr/bin/env python3
"""Prompt Evaluation Pipeline

This script evaluates prompts using the FactSelfCheck framework.
It takes prompts from a CSV file, generates samples, builds knowledge graphs,
and evaluates hallucinations using various prediction methods.

Configuration is loaded from .env file (no command-line arguments needed).
Results are automatically saved to CSV with auto-save functionality.
"""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from langchain_openai import ChatOpenAI
from loguru import logger
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hallucinations_kg.defaults import LANGCHAIN_CACHE_PATH
from hallucinations_kg.metrics import auc_pr
from hallucinations_kg.models.predictor import (
    FactOccurrencePredictor,
    Graph,
    Triple,
)
from hallucinations_kg.utils.config import EvaluationConfig, load_env_variables
from hallucinations_kg.utils.csv_handler import CSVHandler, ResultFormatter


def setup_logging(log_level: str) -> None:
    """Setup logging configuration."""
    logger.remove()  # Remove default handler
    logger.add(
        sys.stdout,
        level=log_level,
        format="<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    )
    logger.add(
        "evaluation.log",
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
        rotation="10 MB",
        retention="3",
    )


def create_llm_client(config: EvaluationConfig) -> ChatOpenAI:
    """
    Create LLM client based on configuration.

    Args:
        config: EvaluationConfig instance

    Returns:
        ChatOpenAI client configured for the specified provider
    """
    llm_config = config.llm_config

    if llm_config.provider == "openai":
        logger.info(f"Creating OpenAI client with model: {llm_config.model}")
        return ChatOpenAI(
            model=llm_config.model,
            api_key=llm_config.api_key,
        )

    elif llm_config.provider in ["selfhosted", "custom"]:
        logger.info(
            f"Creating {llm_config.provider} client at {llm_config.api_url} "
            f"with model: {llm_config.model}"
        )
        return ChatOpenAI(
            model=llm_config.model,
            api_key=llm_config.api_key,
            base_url=llm_config.api_url,
        )

    else:
        raise ValueError(f"Unknown LLM provider: {llm_config.provider}")


def generate_samples(llm: ChatOpenAI, prompt: str, num_samples: int) -> list[str]:
    """
    Generate multiple samples using the LLM.

    Args:
        llm: ChatOpenAI client
        prompt: The prompt to generate samples from
        num_samples: Number of samples to generate

    Returns:
        List of generated samples
    """
    samples = []
    logger.debug(f"Generating {num_samples} samples for prompt: {prompt[:50]}...")

    for i in range(num_samples):
        try:
            response = llm.invoke(prompt)
            samples.append(response.content)
            logger.debug(f"Generated sample {i + 1}/{num_samples}")
        except Exception as e:
            logger.warning(f"Failed to generate sample {i + 1}: {e}")
            continue

    if not samples:
        logger.error(f"Failed to generate any samples for prompt: {prompt[:50]}")
        return []

    logger.debug(f"Successfully generated {len(samples)} samples")
    return samples


def extract_entities_and_relations(
    llm: ChatOpenAI,
    text: str,
) -> list[tuple[str, str, str]]:
    """
    Extract entities and relations from text to build knowledge graph.

    Args:
        llm: ChatOpenAI client
        text: Text to extract from

    Returns:
        List of triples (head, relation, tail)
    """
    extraction_prompt = f"""Extract entities and relations from the following text.
Return as a JSON list of triples in the format: [["entity1", "relation", "entity2"], ...]

Text: {text}

JSON:"""

    try:
        response = llm.invoke(extraction_prompt)
        result_text = response.content

        # Try to extract JSON from response
        start_idx = result_text.find("[")
        end_idx = result_text.rfind("]") + 1
        if start_idx != -1 and end_idx > start_idx:
            json_str = result_text[start_idx:end_idx]
            triples = json.loads(json_str)
            logger.debug(f"Extracted {len(triples)} triples from text")
            return triples
        else:
            logger.warning("Could not find JSON in response")
            return []
    except Exception as e:
        logger.warning(f"Failed to extract relations: {e}")
        return []


def build_knowledge_graphs(
    llm: ChatOpenAI,
    samples: list[str],
) -> list[Graph]:
    """
    Build knowledge graphs from samples.

    Args:
        llm: ChatOpenAI client
        samples: List of sample texts

    Returns:
        List of Graph objects
    """
    graphs = []
    logger.debug(f"Building knowledge graphs from {len(samples)} samples...")

    for sample in samples:
        triples = extract_entities_and_relations(llm, sample)
        if triples:
            graph = Graph.from_triples(triples)
            graphs.append(graph)

    logger.debug(f"Built {len(graphs)} knowledge graphs")
    return graphs


def compute_hallucination_scores(
    graphs: list[Graph],
    predictor: FactOccurrencePredictor,
) -> dict[str, Any]:
    """
    Compute hallucination scores for all facts across all graphs.

    Args:
        graphs: List of knowledge graphs
        predictor: FactOccurrencePredictor instance

    Returns:
        Dictionary containing detailed scores and statistics
    """
    if not graphs:
        return {
            "fact_scores": [],
            "mean_score": 0.0,
            "median_score": 0.0,
            "std_score": 0.0,
            "min_score": 0.0,
            "max_score": 0.0,
            "total_facts": 0,
        }

    all_scores = []
    fact_details = []

    # Extract all facts and compute scores
    for graph_idx, graph in enumerate(graphs):
        for fact_idx, fact in enumerate(graph.triples):
            try:
                score = predictor.predict(fact, graphs)
                all_scores.append(score)
                fact_details.append({
                    "graph_idx": graph_idx,
                    "fact": fact.to_str(),
                    "score": score,
                })
            except Exception as e:
                logger.warning(f"Failed to compute score for fact {fact.to_str()}: {e}")
                continue

    if not all_scores:
        return {
            "fact_scores": [],
            "mean_score": 0.0,
            "median_score": 0.0,
            "std_score": 0.0,
            "min_score": 0.0,
            "max_score": 0.0,
            "total_facts": 0,
        }

    scores_array = np.array(all_scores)

    return {
        "fact_scores": all_scores,  # 所有事实的分数
        "fact_details": fact_details,  # 每个事实的详细信息
        "mean_score": float(np.mean(scores_array)),  # 平均幻觉分数
        "median_score": float(np.median(scores_array)),  # 中位数
        "std_score": float(np.std(scores_array)),  # 标准差
        "min_score": float(np.min(scores_array)),  # 最小值 (最不像幻觉)
        "max_score": float(np.max(scores_array)),  # 最大值 (最像幻觉)
        "total_facts": len(all_scores),  # 总事实数
        "high_hallucination_count": int(np.sum(scores_array > 0.7)),  # 高幻觉数 (>0.7)
        "low_hallucination_count": int(np.sum(scores_array < 0.3)),  # 低幻觉数 (<0.3)
    }


def evaluate_hallucinations(
    prompt: str,
    samples: list[str],
    config: EvaluationConfig,
) -> dict[str, Any]:
    """
    Comprehensive hallucination evaluation using FactSelfCheck framework.

    Args:
        prompt: The original prompt
        samples: List of generated samples
        config: EvaluationConfig instance

    Returns:
        Dictionary containing detailed evaluation results and scores
    """
    result = {
        "prompt": prompt,
        "num_samples": len(samples),
        "timestamp": pd.Timestamp.now().isoformat(),
    }

    if not samples:
        logger.warning(f"No samples to evaluate for prompt: {prompt[:50]}")
        result["error"] = "No samples generated"
        result["status"] = "failed"
        result["hallucination_score"] = None
        return result

    try:
        # Create LLM client
        llm = create_llm_client(config)

        # Build knowledge graphs from samples
        graphs = build_knowledge_graphs(llm, samples)
        result["num_graphs_built"] = len(graphs)

        if not graphs:
            logger.warning(f"No knowledge graphs built for prompt: {prompt[:50]}")
            result["error"] = "Failed to build knowledge graphs"
            result["status"] = "failed"
            result["hallucination_score"] = None
            return result

        # Initialize FactOccurrencePredictor
        # This predictor scores facts based on their frequency across samples
        # High score = low frequency = likely hallucination
        # Low score = high frequency = likely factual
        predictor = FactOccurrencePredictor()

        # Compute hallucination scores for all facts
        score_results = compute_hallucination_scores(graphs, predictor)

        # Add score results to output
        result.update(score_results)

        # 计算最终综合幻觉分数 (0-1)
        # 使用平均分数作为该提示词的总体幻觉评分
        result["hallucination_score"] = score_results["mean_score"]

        result["status"] = "success"
        logger.debug(
            f"Hallucination evaluation completed. "
            f"Mean score: {score_results['mean_score']:.4f}, "
            f"Total facts: {score_results['total_facts']}"
        )

    except Exception as e:
        logger.error(f"Error during hallucination evaluation: {e}")
        result["error"] = str(e)
        result["status"] = "failed"
        result["hallucination_score"] = None

    return result


def process_prompts(config: EvaluationConfig) -> dict[str, float]:
    """
    Main processing function that evaluates prompts from CSV.

    Args:
        config: EvaluationConfig instance

    Returns:
        Dictionary with summary statistics
    """
    logger.info("=" * 80)
    logger.info("Starting Prompt Evaluation Pipeline")
    logger.info("=" * 80)
    logger.info(f"LLM Provider: {config.llm_config.provider}")
    logger.info(f"LLM Model: {config.llm_config.model}")
    logger.info(f"Input CSV: {config.input_csv_path}")
    logger.info(f"Output CSV: {config.output_csv_path}")
    logger.info(f"Auto-save interval: {config.auto_save_interval}")
    logger.info(f"Number of samples per prompt: {config.num_samples}")

    # Create LLM client
    llm = create_llm_client(config)

    # Initialize CSV handler
    all_hallucination_scores = []
    successful_evaluations = 0
    failed_evaluations = 0

    with CSVHandler(
        input_path=config.input_csv_path,
        output_path=config.output_csv_path,
        auto_save_interval=config.auto_save_interval,
    ) as csv_handler:
        # Get unprocessed prompts
        unprocessed_prompts = csv_handler.get_unprocessed_prompts()

        if not unprocessed_prompts:
            logger.info("All prompts have been processed!")
            return {
                "total_prompts": 0,
                "successful": 0,
                "failed": 0,
                "overall_hallucination_score": None,
            }

        # Process each prompt
        progress_bar = tqdm(
            unprocessed_prompts,
            desc="Evaluating prompts",
            unit="prompt",
        )

        for prompt in progress_bar:
            try:
                # Generate samples
                samples = generate_samples(llm, prompt, config.num_samples)

                # Evaluate hallucinations
                result = evaluate_hallucinations(prompt, samples, config)

                # Track scores for final statistics
                if result["status"] == "success" and result.get("hallucination_score") is not None:
                    all_hallucination_scores.append(result["hallucination_score"])
                    successful_evaluations += 1
                else:
                    failed_evaluations += 1

                # Add to CSV
                result_formatted = ResultFormatter.flatten_result(result)
                csv_handler.add_result(prompt, result_formatted)

                progress_bar.set_postfix({
                    "status": result.get("status", "unknown"),
                    "score": f"{result.get('hallucination_score', 'N/A'):.3f}" if result.get("hallucination_score") is not None else "N/A",
                })

            except Exception as e:
                logger.error(f"Failed to process prompt '{prompt[:50]}...': {e}")
                error_result = {
                    "prompt": prompt,
                    "error": str(e),
                    "status": "failed",
                    "hallucination_score": None,
                }
                csv_handler.add_result(prompt, error_result)
                failed_evaluations += 1
                continue

        # Save any remaining data
        csv_handler.save()

    # Calculate overall statistics
    overall_hallucination_score = None
    if all_hallucination_scores:
        overall_hallucination_score = float(np.mean(all_hallucination_scores))

    summary = {
        "total_prompts": len(unprocessed_prompts),
        "successful": successful_evaluations,
        "failed": failed_evaluations,
        "overall_hallucination_score": overall_hallucination_score,
        "std_hallucination_score": float(np.std(all_hallucination_scores)) if all_hallucination_scores else None,
    }

    logger.info("=" * 80)
    logger.info("Evaluation Pipeline Completed")
    logger.info("=" * 80)
    logger.info(f"Total prompts processed: {summary['total_prompts']}")
    logger.info(f"Successful evaluations: {summary['successful']}")
    logger.info(f"Failed evaluations: {summary['failed']}")
    logger.info(f"Overall hallucination score: {summary['overall_hallucination_score']:.4f}" if summary['overall_hallucination_score'] is not None else "N/A")
    logger.info(f"Results saved to: {config.output_csv_path}")

    return summary


def main() -> None:
    """Main entry point."""
    try:
        # Load configuration from environment
        config = EvaluationConfig.from_env()
        config.validate()

        # Setup logging
        setup_logging(config.log_level)

        logger.info("Configuration loaded successfully")
        logger.debug(f"Config: {config}")

        # Run evaluation pipeline
        summary = process_prompts(config)

        logger.info("Pipeline finished successfully")
        logger.info(f"Summary: {summary}")
        sys.exit(0)

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
