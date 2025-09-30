# FactSelfCheck

This repository contains the code used in the research paper titled
[**"FactSelfCheck: Fact-Level Black-Box Hallucination Detection for LLMs"**](https://arxiv.org/abs/2503.17229)
authored by Albert Sawczyn, Jakub Binkowski, Denis Janiak, Bogdan Gabrys, Tomasz Kajdanowicz.

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
* OpenAI API key

### Environment variables

We use a few environment variables to configure the project. You can set them manually or put them in `.env` file. The file is loaded automatically without overriding existing variables.

The example of the `.env` file:

```bash
# OpenAI key
OPENAI_API_KEY=[your-openai-key]

# Self-hosted OpenAI-compatible server
SELFHOSTED_API_URL=[your-api-url]
SELFHOSTED_API_KEY=[your-api-key]
```

### Installing

To install all dependencies, run:

```bash
pip install -r requirements.txt
```

### Downloading data

The repository uses DVC to manage the data. To download the data, run:

```bash
dvc pull
```

[!NOTE]
The data will be available soon.

## Reproducing

### DVC

The repository uses DVC to manage the dataset construction pipeline.

* `dvc.yaml` contains all of the stages except notebooks with results of experiments.

To reproduce all dvc stages run:

```bash
dvc repro
```

### Notebooks

Notebooks with results of experiments are available in the `notebooks` directory.

### Cache

The repository uses LangChain cache to store the results of the LLM calls. The cache is stored in the `.langchain.db` file. To clear the cache remove the file.
