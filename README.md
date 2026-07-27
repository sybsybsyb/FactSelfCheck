# FactSelfCheck

This repository contains the code used in the research paper titled [**"FactSelfCheck: Fact-Level Black-Box Hallucination Detection for LLMs"**](https://arxiv.org/abs/2503.17229) authored by Albert Sawczyn, Jakub Binkowski, Denis Janiak, Bogdan Gabrys, and Tomasz Kajdanowicz.

## Citation

If you use this repository in your work, please cite it as follows:

```bibtex
@misc{sawczyn2025factselfcheckfactlevelblackboxhallucination,
      title={FactSelfCheck: Fact-Level Black-Box Hallucination Detection for LLMs},
      author={Albert Sawczyn and Jakub Binkowski and Denis Janiak and Bogdan Gabrys and Tomasz Kajdanowicz},
      year={2025},
      eprint={2503.17229},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2503.17229},
}
```

## License

CC BY-SA 4.0

## FavaMultiSamples

The FavaMultiSamples dataset is available at [Hugging Face](https://huggingface.co/datasets/graphml-lab-pwr/FavaMultiSamples).

## Getting Started

### Prerequisites

* Configured Python 3.12 environment (e.g. using conda)
* Self-hosted OpenAI-compatible server with Llama-3.1-70B-Instruct model. We used VLLM API (see [VLLM](https://github.com/vllm-project/vllm)).
* OpenAI API key (for OpenAI provider)

### Installing

To install all dependencies, run:

```bash
pip install -r requirements.txt
```

### Environment Configuration

We use environment variables to configure the project. You can set them manually or put them in a `.env` file. The file is loaded automatically without overriding existing variables.

Copy the example configuration file:

```bash
cp .env.example .env
```

Then edit `.env` with your configuration:

```bash
# OpenAI Configuration
OPENAI_API_KEY=your-openai-api-key-here

# Self-hosted OpenAI-compatible Server Configuration
SELFHOSTED_API_URL=http://localhost:8000/v1
SELFHOSTED_API_KEY=your-selfhosted-api-key-here

# LLM Configuration for Evaluation
# Choose which LLM provider to use: "openai", "selfhosted", or "custom"
LLM_PROVIDER=openai

# Model selection
LLM_MODEL=gpt-4

# Input/Output Configuration
INPUT_CSV_PATH=data/prompts.csv
OUTPUT_CSV_PATH=data/results/evaluation_results.csv

# Auto-save interval (number of items before saving)
AUTO_SAVE_INTERVAL=10

# Number of samples to generate per prompt
NUM_SAMPLES=10

# Random seed for reproducibility
RANDOM_SEED=42

# Logging level: DEBUG, INFO, WARNING, ERROR
LOG_LEVEL=INFO
```

### Downloading Data

The repository uses DVC to manage the data. To download the data, run:

```bash
dvc pull
```

[!NOTE]
The data will be available soon.

## Usage

### Prompt Evaluation Pipeline

The prompt evaluation pipeline allows you to evaluate prompts using the FactSelfCheck framework. It:
- Reads prompts from a CSV file
- Generates multiple samples for each prompt using your configured LLM
- Builds knowledge graphs from the samples
- Evaluates hallucinations using the FactSelfCheck framework
- Saves results to a CSV file with automatic checkpointing

#### Quick Start

1. **Prepare your input CSV file** with a `prompt` column:
   ```csv
   prompt
   "What is the capital of France?"
   "Tell me about machine learning"
   "Describe quantum computing"
   ```

2. **Configure your environment** in `.env`:
   ```bash
   INPUT_CSV_PATH=data/prompts.csv
   OUTPUT_CSV_PATH=data/results/evaluation_results.csv
   LLM_PROVIDER=openai
   LLM_MODEL=gpt-4
   ```

3. **Run the evaluation pipeline**:
   ```bash
   python scripts/evaluate_prompts.py
   ```

#### Features

- **No Command-Line Arguments Required**: All configuration comes from the `.env` file
- **Multiple LLM Providers**: Support for OpenAI, self-hosted servers, or custom LLM endpoints
- **Automatic Checkpointing**: Results are automatically saved at regular intervals, preventing data loss if the server crashes
- **Resume from Interruption**: If the pipeline is interrupted, it will resume from where it left off, only processing prompts that haven't been evaluated yet
- **Comprehensive Logging**: Detailed logs are saved to `evaluation.log` with both stdout and file output
- **Progress Tracking**: Real-time progress bar showing evaluation status

#### LLM Provider Configuration

**OpenAI:**
```bash
LLM_PROVIDER=openai
LLM_MODEL=gpt-4
OPENAI_API_KEY=sk-...
```

**Self-hosted (e.g., VLLM):**
```bash
LLM_PROVIDER=selfhosted
LLM_MODEL=llama-3.1-70b-instruct
SELFHOSTED_API_URL=http://localhost:8000/v1
SELFHOSTED_API_KEY=your-key
```

**Custom LLM:**
```bash
LLM_PROVIDER=custom
CUSTOM_LLM_API_URL=https://your-llm-api.com/v1
CUSTOM_LLM_API_KEY=your-api-key
CUSTOM_LLM_MODEL=your-model-name
```

#### Output Format

The output CSV file contains:
- `prompt`: The original prompt
- `num_samples`: Number of samples generated
- `num_graphs_built`: Number of knowledge graphs successfully built
- `status`: Evaluation status (success/failed)
- `timestamp`: Timestamp of evaluation
- `error`: Error message (if status is failed)

Example output:
```csv
prompt,num_samples,num_graphs_built,status,timestamp,error
"What is AI?",10,8,success,2026-07-27T16:00:00.000000,
"Tell me about xyz",10,0,failed,2026-07-27T16:01:00.000000,"Failed to extract relations"
```

#### Auto-Save Functionality

The pipeline automatically saves results every N prompts (configurable via `AUTO_SAVE_INTERVAL`). This ensures that if the server crashes or the process is interrupted, no progress is lost. The pipeline will automatically detect previously processed prompts and continue from where it left off.

To disable auto-save (save only at the end):
```bash
AUTO_SAVE_INTERVAL=0
```

## Reproducing Experiments

### DVC Pipeline

The repository uses DVC to manage the dataset construction pipeline.

* `dvc.yaml` contains all of the stages except notebooks with results of experiments.

To reproduce all DVC stages run:

```bash
dvc repro
```

### Notebooks

Notebooks with results of experiments are available in the `notebooks` directory.

### Cache

The repository uses LangChain cache to store the results of the LLM calls. The cache is stored in the `.langchain.db` file. To clear the cache, remove the file:

```bash
rm .langchain.db
```

## Project Structure

```
FactSelfCheck/
├── .env.example                 # Example environment configuration
├── .env                         # Local environment configuration (not in git)
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── scripts/
│   ├── evaluate_prompts.py     # Main prompt evaluation pipeline
│   ├── build_graphs_csv.py     # Build knowledge graphs from CSV
│   ├── extract_entities_relations.py
│   ├── fact_text_prompt_answers.py
│   └── ...
├── hallucinations_kg/
│   ├── utils/
│   │   ├── config.py           # Configuration management
│   │   ├── csv_handler.py      # CSV I/O and auto-save
│   │   └── __init__.py
│   ├── models/                  # Prediction models
│   ├── metrics/                 # Evaluation metrics
│   ├── prediction/              # Prediction pipeline
│   └── ...
├── data/
│   ├── prompts.csv             # Input prompts file
│   └── results/
│       └── evaluation_results.csv  # Output results file
├── notebooks/                   # Jupyter notebooks with experiments
├── dvc.yaml                     # DVC pipeline configuration
└── dvc.lock                     # DVC lock file
```

## Troubleshooting

### "Input CSV file not found"
Ensure the `INPUT_CSV_PATH` in your `.env` file points to an existing CSV file with a `prompt` column.

### "OPENAI_API_KEY not set in environment"
Make sure your `.env` file contains a valid `OPENAI_API_KEY` or set it in your shell environment.

### LLM Connection Errors
- For self-hosted servers, ensure the server is running and accessible at the specified URL
- Check that the API key and model name are correct
- Test connectivity: `curl -H "Authorization: Bearer YOUR_KEY" http://your-api-url/v1/models`

### Results Not Saving
- Check that the output directory exists and is writable
- Ensure `AUTO_SAVE_INTERVAL` is set to a positive number
- Check the `evaluation.log` file for detailed error messages

### Resuming Interrupted Pipeline
The pipeline will automatically detect processed prompts and continue. To start fresh:
1. Delete the output CSV file
2. Or rename it to create a new output file

## Contributing

Contributions are welcome! Please follow the existing code style and ensure all tests pass before submitting a pull request.

## Support

For issues, questions, or suggestions, please open an issue on GitHub.
