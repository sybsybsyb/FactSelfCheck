import os
from pathlib import Path

from tenacity import retry, stop_after_attempt, wait_exponential

ROOT_PATH = Path(os.path.dirname(__file__)).parent.absolute()
PLOTS_PATH = ROOT_PATH / "data/results/plots"


LANGCHAIN_CACHE_PATH = ROOT_PATH / ".langchain.db"
LLM_RETRIES = 10

retry_llm_call = retry(
    stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=300)
)
