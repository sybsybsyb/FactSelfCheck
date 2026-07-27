import json
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger


class CSVHandler:
    """Handle CSV file operations with auto-save functionality."""

    def __init__(
        self,
        input_path: str,
        output_path: str,
        auto_save_interval: int = 10,
    ):
        """
        Initialize CSVHandler.

        Args:
            input_path: Path to input CSV file with prompts
            output_path: Path to output CSV file for results
            auto_save_interval: Number of rows before auto-saving (0 to disable)
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.auto_save_interval = auto_save_interval
        self.processed_count = 0
        self.data = []

        # Load input CSV
        self._load_input_csv()

        # Initialize output CSV if it doesn't exist
        self._initialize_output_csv()

        logger.info(f"CSVHandler initialized with {len(self.input_data)} rows from {input_path}")
        logger.info(f"Output will be saved to {output_path}")
        logger.info(f"Auto-save interval: {auto_save_interval} rows")

    def _load_input_csv(self) -> None:
        """Load input CSV file."""
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input CSV file not found: {self.input_path}")

        try:
            self.input_data = pd.read_csv(self.input_path)
            if "prompt" not in self.input_data.columns:
                raise ValueError("Input CSV must have a 'prompt' column")
            logger.info(f"Loaded {len(self.input_data)} prompts from {self.input_path}")
        except Exception as e:
            logger.error(f"Failed to load input CSV: {e}")
            raise

    def _initialize_output_csv(self) -> None:
        """Initialize output CSV file."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # If output file exists, check if it's complete or needs continuation
        if self.output_path.exists():
            try:
                existing_data = pd.read_csv(self.output_path)
                self.processed_count = len(existing_data)
                logger.info(
                    f"Found existing output file with {self.processed_count} processed rows. "
                    "Will append to it."
                )
            except Exception as e:
                logger.warning(f"Could not read existing output file: {e}. Starting fresh.")
                self.processed_count = 0
        else:
            self.processed_count = 0
            logger.info(f"Output file does not exist. Will create new file at {self.output_path}")

    def add_result(
        self,
        prompt: str,
        result: dict[str, Any],
        batch_save: bool = False,
    ) -> None:
        """
        Add a result row to the output data.

        Args:
            prompt: The original prompt
            result: Dictionary containing evaluation results
            batch_save: Force save immediately if True
        """
        row = {"prompt": prompt, **result}
        self.data.append(row)
        self.processed_count += 1

        # Auto-save if interval is reached
        if (
            self.auto_save_interval > 0
            and self.processed_count % self.auto_save_interval == 0
        ) or batch_save:
            self.save()

    def save(self) -> None:
        """Save accumulated results to CSV file."""
        if not self.data:
            logger.debug("No new data to save")
            return

        try:
            df_new = pd.DataFrame(self.data)

            # Check if output file exists and append or create new
            if self.output_path.exists():
                df_existing = pd.read_csv(self.output_path)
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            else:
                df_combined = df_new

            df_combined.to_csv(self.output_path, index=False)
            logger.info(
                f"Saved {len(self.data)} new results to {self.output_path} "
                f"(total: {len(df_combined)} rows)"
            )
            self.data = []  # Clear buffer after saving

        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise

    def get_unprocessed_prompts(self) -> list[str]:
        """
        Get list of unprocessed prompts.

        Returns:
            List of prompts that haven't been processed yet
        """
        if self.output_path.exists():
            try:
                df_output = pd.read_csv(self.output_path)
                processed_prompts = set(df_output["prompt"].values)
                unprocessed = self.input_data[
                    ~self.input_data["prompt"].isin(processed_prompts)
                ]["prompt"].tolist()
                logger.info(
                    f"Found {len(unprocessed)} unprocessed prompts "
                    f"({len(processed_prompts)} already processed)"
                )
                return unprocessed
            except Exception as e:
                logger.warning(f"Could not determine processed prompts: {e}")
                return self.input_data["prompt"].tolist()
        else:
            logger.info(f"No output file found. Will process all {len(self.input_data)} prompts")
            return self.input_data["prompt"].tolist()

    def close(self) -> None:
        """Close handler and save any remaining data."""
        if self.data:
            logger.info("Saving remaining data before closing...")
            self.save()
        logger.info("CSVHandler closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        if exc_type is not None:
            logger.error(f"Error occurred: {exc_type.__name__}: {exc_val}")
            return False
        return True


class ResultFormatter:
    """Format and structure evaluation results."""

    @staticmethod
    def flatten_result(result: dict[str, Any]) -> dict[str, Any]:
        """
        Flatten nested result dictionary for CSV storage.

        Args:
            result: Dictionary that may contain nested structures

        Returns:
            Flattened dictionary suitable for CSV storage
        """
        flattened = {}
        for key, value in result.items():
            if isinstance(value, (dict, list)):
                # Convert complex types to JSON strings
                flattened[key] = json.dumps(value)
            else:
                flattened[key] = value
        return flattened

    @staticmethod
    def unflatten_result(row: dict[str, Any], json_columns: list[str] | None = None) -> dict[str, Any]:
        """
        Restore nested structures from flattened CSV data.

        Args:
            row: Dictionary from CSV row
            json_columns: List of column names that contain JSON strings

        Returns:
            Dictionary with restored structures
        """
        if json_columns is None:
            json_columns = []

        result = {}
        for key, value in row.items():
            if key in json_columns and isinstance(value, str):
                try:
                    result[key] = json.loads(value)
                except json.JSONDecodeError:
                    result[key] = value
            else:
                result[key] = value
        return result
