from tenacity import retry, stop_after_attempt, wait_fixed

from hallucinations_kg.utils.langchain import BaseAgent


class FactAnnotationAgent(BaseAgent):
    @retry(stop=stop_after_attempt(10), wait=wait_fixed(1))
    def annotate(
        self,
        triple: tuple[str, str, str],
        source: str,
    ) -> str:
        assert isinstance(triple, tuple)
        assert isinstance(triple[0], str)
        assert isinstance(source, str)
        triple_str = f"('{triple[0]}', '{triple[1]}', '{triple[2]}')"
        result = self.chain.invoke({"fact": triple_str, "source": source}).lower()
        assert isinstance(result, str)
        return result
