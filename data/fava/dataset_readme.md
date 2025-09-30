# FavaMultiSamples

This repository contains the FavaMultiSamples dataset proposed and used in the research paper titled
**"FactSelfCheck: Fact-Level Black-Box Hallucination Detection for LLMs".**

## Citation

If you use this repository in your work, please cite it as follows:

```bibtex
TBA
```

## License

CC BY-SA 4.0


## Dataset structure

The dataset contains all columns from the original Fava dataset, plus the following columns:

| Field Name | Description |
|------------|-------------|
| sentences | List of individual sentences from the generated text. |
| sentences_annotations | List of sentence-level annotations. Single annotation is a list of hallucination types that are present in the sentence.|
| sentences_binary_annotations | List of binary flags (0/1) for each sentence. 1 means the sentence contains at least one type of hallucination, 0 otherwise.|
| text_samples | List of generated text samples. |
| text_samples_model_version | Model version used for generating text samples. |
| text_samples_generation_seeds | List of integers used as random seeds for text generation. |
| text_samples_openai_system_fingerprint | System configuration identifier (null for llama). |
| text_samples_stop_reason | List of generation termination reasons for the llama model ("stop" means normal completion). |
