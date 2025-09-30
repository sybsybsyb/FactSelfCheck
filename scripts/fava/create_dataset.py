import re
from pathlib import Path
from typing import Any, Generator

import typer
from bs4 import BeautifulSoup
from datasets import load_dataset
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableSerializable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt

TAGS_ANNOTATION = ["entity", "relation", "contradictory", "invented", "subjective", "unverifiable"]
TAGS_UNWANTED = ["other", "format", "delete", "fictional", "mark"]

COLUMNS_TO_REMOVE = ["num_tokens"]
RENAME_COLUMNS = {
    "model_version": "text_samples_model_version",
    "generation_seeds": "text_samples_generation_seeds",
    "openai_system_fingerprint": "text_samples_openai_system_fingerprint",
    "stop_reason": "text_samples_stop_reason",
}

COLUMNS_ORDER = [
    "prompt",
    "output",
    "annotated",
    "subject",
    "dataset",
    "model",
    "sentences",
    "sentences_annotations",
    "sentences_binary_annotations",
    "text_samples",
    "text_samples_model_version",
    "text_samples_generation_seeds",
    "text_samples_openai_system_fingerprint",
    "text_samples_stop_reason",
]

MODEL_VERSIONS = {
    "chatgpt": "gpt-3.5-turbo-1106",
    "llama": "meta-llama/Llama-2-70b-chat-hf",
}

set_llm_cache(SQLiteCache(database_path=".langchain.db"))


class SentenceSplitterAgent:
    class _Result(BaseModel):
        sentences: list[str]

    def __init__(self, llm: ChatOpenAI) -> None:
        prompt = self._get_prompt()
        self.chain = self._get_chain(prompt, llm)

    def _get_chain(self, prompt: ChatPromptTemplate, llm: ChatOpenAI) -> RunnableSerializable:
        structured_llm = llm.with_structured_output(self._Result, method="json_schema")
        return prompt | structured_llm

    @retry(stop=stop_after_attempt(5))
    def split(self, text: str) -> Generator[tuple[str, tuple[int, int]], None, None]:
        result = self.chain.invoke({"text": text})
        if isinstance(result, dict):  # cached result is a dict
            result = self._Result(**result)
        for sentence in result.sentences:
            sentence_start, sentence_end = self._find_ignoring_whitespace(text, sentence)
            assert sentence_start != -1
            yield text[sentence_start:sentence_end], (sentence_start, sentence_end)

    def _get_prompt(self) -> ChatPromptTemplate:
        human_prompt_template = PromptTemplate(
            template="Split the following text into sentences. Do not change the text, preserve whitespaces ('\n', '\t', ' ', etc.), even if they are doubled. Text: {text}",
            input_variables=["text"],
        )
        human_message_prompt = HumanMessagePromptTemplate(prompt=human_prompt_template)
        chat_prompt = ChatPromptTemplate.from_messages([human_message_prompt])
        return chat_prompt

    def _find_ignoring_whitespace(self, text: str, sentence: str) -> tuple[int, int]:
        # Remove all whitespace for comparison
        sentence_clean = re.sub(r"\s+", "", sentence)

        # Build a cleaned version of text while mapping indices
        clean_chars = []
        index_map = []  # Maps positions in clean_chars back to text

        for i, c in enumerate(text):
            if not c.isspace():
                clean_chars.append(c)
                index_map.append(i)

        clean_text = "".join(clean_chars)

        # Search for cleaned sentence in cleaned text
        idx = clean_text.find(sentence_clean)

        if idx == -1:
            return -1, -1

        start = index_map[idx]
        end = index_map[idx + len(sentence_clean) - 1] + 1  # +1 to make end exclusive
        return start, end


def main(
    input_file: Path = typer.Option("data/fava/fava_with_samples.json"),
    output_file: Path = typer.Option("data/fava-sampling/"),
) -> None:
    dataset = load_dataset("json", data_files={"test": str(input_file)})

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    sentence_splitter = SentenceSplitterAgent(llm)

    with get_openai_callback() as cb:
        dataset = dataset.map(
            get_sentences_with_annotations, fn_kwargs={"sentence_splitter": sentence_splitter}
        )
        print(cb)

    dataset = dataset.map(add_binary_annotations)
    dataset = dataset.map(lambda x: {"model_version": MODEL_VERSIONS[x["model"]]})
    dataset = dataset.remove_columns(COLUMNS_TO_REMOVE)
    dataset = dataset.rename_columns(RENAME_COLUMNS)
    dataset = dataset.select_columns(COLUMNS_ORDER)

    dataset.save_to_disk(output_file)


def get_sentences_with_annotations(
    entry: dict[str, Any], sentence_splitter: SentenceSplitterAgent
) -> dict[str, Any]:
    annotated_text = entry["annotated"]
    text_without_unwanted_tags = remove_unwanted_tags(annotated_text)
    clean_text, tags_positions = remove_all_tags(text_without_unwanted_tags)
    tags_spans = list(find_tags_spans(text_without_unwanted_tags, clean_text, tags_positions))

    sentence_tags: list[list[str]] = []
    sentences: list[str] = []
    for sentence, (s_start, s_end) in sentence_splitter.split(clean_text):
        sentences.append(sentence)
        sentence_tags.append([])
        for tag, start_without_tags, end_without_tags in tags_spans:
            if (
                s_start <= start_without_tags < s_end
                or s_start < end_without_tags <= s_end
                or (start_without_tags <= s_start and end_without_tags >= s_end)
            ):
                if tag not in sentence_tags[-1]:
                    sentence_tags[-1].append(tag)

    flattened_annotations = [a for s in sentence_tags for a in s]
    for tag in TAGS_ANNOTATION:
        if entry["annotated"].count(f"<{tag}>") > 0:
            assert tag in flattened_annotations

    return {"sentences": sentences, "sentences_annotations": sentence_tags}


def add_binary_annotations(entry: dict[str, Any]) -> dict[str, Any]:
    binary_annotations = [1 if len(s) > 0 else 0 for s in entry["sentences_annotations"]]
    return {"sentences_binary_annotations": binary_annotations}


def remove_tags(text: str, tags: list[str]) -> str:
    for t in tags:
        text = text.replace(f"<{t}>", "").replace(f"</{t}>", "")
    return text


def remove_unwanted_tags(text: str) -> str:
    text = re.sub(r"<mark>.*?<\/mark>", "", text)
    assert text.count("<mark>") == 0
    return remove_tags(text, TAGS_UNWANTED)


def remove_all_tags(text: str) -> tuple[str, dict[str, list[int]]]:
    tag_pattern = re.compile(r"<([^>]+)>")
    positions: dict[str, list[int]] = {}
    cleaned = ""
    last_end = 0
    for match in tag_pattern.finditer(text):
        tag_name = match.group(1)
        start, end = match.span()
        cleaned += text[last_end:start]
        pos = len(cleaned)
        if tag_name not in positions:
            positions[tag_name] = []
        positions[tag_name].append(pos)
        last_end = end
    cleaned += text[last_end:]
    assert not re.search(r"<[^>]+>", cleaned), "String contains HTML/XML-like tags!"
    return cleaned, positions


def find_tags_spans(
    text: str, text_without_tags: str, tags_positions: dict[str, list[int]]
) -> Generator[tuple[str, int, int], None, None]:
    soup = BeautifulSoup(text, "html.parser")
    for tag in TAGS_ANNOTATION:
        for match in soup.find_all(tag):
            if text_without_tags.count(match.text) != 1:
                pattern = re.escape(match.text)
                match_starts = [m.start() for m in re.finditer(pattern, text_without_tags)]
                tag_starts = tags_positions[tag]
                common_starts = [s for s in match_starts if s in tag_starts]
                for common_start in common_starts:
                    end_without_tags = common_start + len(match.text)
                    yield tag, common_start, end_without_tags
            else:
                assert (
                    text_without_tags.count(match.text) == 1
                ), f"Tag {tag} found {text_without_tags.count(match.text)} times in text_without_tags: {text_without_tags}"
                start_without_tags = text_without_tags.find(match.text)
                end_without_tags = start_without_tags + len(match.text)
                yield tag, start_without_tags, end_without_tags


if __name__ == "__main__":
    typer.run(main)
