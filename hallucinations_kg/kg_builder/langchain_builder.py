from asyncio import Semaphore

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from tqdm.asyncio import tqdm

from hallucinations_kg.kg_builder.llm_graph_tranformer import CSVLLMGraphTransformer


class CSVLangChainKGBuilder:
    def __init__(
        self,
        llm: BaseChatModel,
        prompt: ChatPromptTemplate,
    ) -> None:
        self.llm = llm
        self.prompt = prompt

    async def abuild_sentence(
        self,
        sentences: list[list[str]],
        full_text: list[str],
        allowed_nodes: list[list[str]],
        allowed_relationships: list[list[str]],
        n_jobs: int,
    ) -> list[list[str]]:
        with tqdm(
            total=sum(len(sentences) for sentences in sentences), desc="Converting to graphs"
        ) as progress_bar:
            sem = Semaphore(n_jobs)
            llm_transformer = self.get_llm_transformer()

            async def sem_task(
                task_sentences: list[str], t: str, an: list[str], ar: list[str], sem: Semaphore
            ) -> list[str]:
                assert isinstance(task_sentences, list)
                async with sem:
                    results = []
                    for s in task_sentences:
                        results.append(await llm_transformer.aprocess_sentence(s, t, an, ar))
                        progress_bar.update(1)
                    return results

            tasks = [
                sem_task(sentences_, t, an, ar, sem)
                for sentences_, t, an, ar in zip(
                    sentences, full_text, allowed_nodes, allowed_relationships, strict=True
                )
            ]
            return await tqdm.gather(*tasks, desc="Converting to graphs", disable=True)

    async def abuild_samples(
        self,
        samples: list[list[str]],
        allowed_nodes: list[list[str]],
        allowed_relationships: list[list[str]],
        n_jobs: int,
    ) -> list[list[str]]:
        sem = Semaphore(n_jobs)
        llm_transformer = self.get_llm_transformer()
        with tqdm(
            total=sum(len(samples) for samples in samples), desc="Converting to graphs"
        ) as progress_bar:

            async def sem_task(
                samples: list[str], an: list[str], ar: list[str], sem: Semaphore
            ) -> list[str]:
                async with sem:
                    results = []
                    for sample in tqdm(samples, disable=True):
                        results.append(await llm_transformer.aprocess_doc(sample, an, ar))
                        progress_bar.update(1)
                    return results

            tasks = [
                sem_task(samples_, an, ar, sem)
                for samples_, an, ar in zip(
                    samples, allowed_nodes, allowed_relationships, strict=True
                )
            ]
            return await tqdm.gather(*tasks, desc="Converting to graphs", disable=True)

    def get_llm_transformer(self) -> CSVLLMGraphTransformer:
        return CSVLLMGraphTransformer(
            llm=self.llm,
            prompt=self.prompt,
        )
