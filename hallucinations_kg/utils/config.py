import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger


def load_env_variables() -> None:
    """Load environment variables from .env file if it exists."""
    env_path = Path.cwd() / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        logger.warning(f".env file not found at {env_path}. Using system environment variables.")


@dataclass
class LLMConfig:
    """Configuration for LLM provider."""
    provider: str  # "openai", "selfhosted", or "custom"
    api_url: str | None = None
    api_key: str | None = None
    model: str = "gpt-4"

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Create LLMConfig from environment variables."""
        provider = os.getenv("LLM_PROVIDER", "openai").lower()
        model = os.getenv("LLM_MODEL", "gpt-4")

        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set in environment")
            return cls(provider="openai", api_key=api_key, model=model)

        elif provider == "selfhosted":
            api_url = os.getenv("SELFHOSTED_API_URL")
            api_key = os.getenv("SELFHOSTED_API_KEY")
            if not api_url or not api_key:
                raise ValueError("SELFHOSTED_API_URL and SELFHOSTED_API_KEY must be set")
            return cls(provider="selfhosted", api_url=api_url, api_key=api_key, model=model)

        elif provider == "custom":
            api_url = os.getenv("CUSTOM_LLM_API_URL")
            api_key = os.getenv("CUSTOM_LLM_API_KEY")
            custom_model = os.getenv("CUSTOM_LLM_MODEL")
            if not api_url or not api_key or not custom_model:
                raise ValueError(
                    "CUSTOM_LLM_API_URL, CUSTOM_LLM_API_KEY, and CUSTOM_LLM_MODEL must be set"
                )
            return cls(provider="custom", api_url=api_url, api_key=api_key, model=custom_model)

        else:
            raise ValueError(f"Unknown LLM_PROVIDER: {provider}")


@dataclass
class EvaluationConfig:
    """Configuration for evaluation pipeline."""
    input_csv_path: str
    output_csv_path: str
    auto_save_interval: int = 10
    num_samples: int = 10
    random_seed: int = 42
    log_level: str = "INFO"
    llm_config: LLMConfig | None = None

    @classmethod
    def from_env(cls) -> "EvaluationConfig":
        """Create EvaluationConfig from environment variables."""
        load_env_variables()

        input_csv = os.getenv("INPUT_CSV_PATH", "data/prompts.csv")
        output_csv = os.getenv("OUTPUT_CSV_PATH", "data/results/evaluation_results.csv")
        auto_save = int(os.getenv("AUTO_SAVE_INTERVAL", "10"))
        num_samples = int(os.getenv("NUM_SAMPLES", "10"))
        random_seed = int(os.getenv("RANDOM_SEED", "42"))
        log_level = os.getenv("LOG_LEVEL", "INFO")

        # Validate paths
        if not Path(input_csv).exists():
            raise FileNotFoundError(f"Input CSV file not found: {input_csv}")

        # Create output directory if it doesn't exist
        output_dir = Path(output_csv).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        llm_config = LLMConfig.from_env()

        return cls(
            input_csv_path=input_csv,
            output_csv_path=output_csv,
            auto_save_interval=auto_save,
            num_samples=num_samples,
            random_seed=random_seed,
            log_level=log_level,
            llm_config=llm_config,
        )

    def validate(self) -> None:
        """Validate configuration."""
        if self.auto_save_interval < 0:
            raise ValueError("AUTO_SAVE_INTERVAL must be >= 0")
        if self.num_samples <= 0:
            raise ValueError("NUM_SAMPLES must be > 0")
        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            raise ValueError(f"Invalid LOG_LEVEL: {self.log_level}")
