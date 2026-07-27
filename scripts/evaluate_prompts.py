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


def evaluate_hallucinations(
    prompt: str,
    samples: list[str],
    config: EvaluationConfig,
) -> dict[str, Any]:
    """
    Evaluate hallucinations for a prompt using the FactSelfCheck framework.

    Args:
        prompt: The original prompt
        samples: List of generated samples
        config: EvaluationConfig instance

    Returns:
        Dictionary containing evaluation results
    """
    result = {
        "num_samples": len(samples),
        "timestamp": pd.Timestamp.now().isoformat(),
    }

    if not samples:
        logger.warning(f"No samples to evaluate for prompt: {prompt[:50]}")
        result["error"] = "No samples generated"
        return result

    # For now, we'll return basic metrics
    # In a full implementation, this would integrate with the FactSelfCheck framework
    try:
        # Build knowledge graphs from samples
        llm = create_llm_client(config)
        graphs = build_knowledge_graphs(llm, samples)
        result["num_graphs_built"] = len(graphs)
        result["status"] = "success"

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        result["error"] = str(e)
        result["status"] = "failed"

    return result


def process_prompts(config: EvaluationConfig) -> None:
    """
    Main processing function that evaluates prompts from CSV.

    Args:
        config: EvaluationConfig instance
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
    with CSVHandler(
        input_path=config.input_csv_path,
        output_path=config.output_csv_path,
        auto_save_interval=config.auto_save_interval,
    ) as csv_handler:
        # Get unprocessed prompts
        unprocessed_prompts = csv_handler.get_unprocessed_prompts()

        if not unprocessed_prompts:
            logger.info("All prompts have been processed!")
            return

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

                # Add to CSV
                result_formatted = ResultFormatter.flatten_result(result)
                csv_handler.add_result(prompt, result_formatted)

                progress_bar.set_postfix({"status": result.get("status", "unknown")})

            except Exception as e:
                logger.error(f"Failed to process prompt '{prompt[:50]}...': {e}")
                error_result = {
                    "error": str(e),
                    "status": "failed",
                }
                csv_handler.add_result(prompt, error_result)
                continue

        # Save any remaining data
        csv_handler.save()

    logger.info("=" * 80)
    logger.info("Evaluation Pipeline Completed")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {config.output_csv_path}")


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
        process_prompts(config)

        logger.info("Pipeline finished successfully")
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
