from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnableSerializable
from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_fixed


class CorrectorAgent:
    def __init__(self, prompt: ChatPromptTemplate, llm: ChatOpenAI):
        self.chain = self._get_chain(prompt, llm)

    def _get_chain(self, prompt: ChatPromptTemplate, llm: ChatOpenAI) -> RunnableSerializable:
        return prompt | llm | StrOutputParser()

    @retry(stop=stop_after_attempt(10), wait=wait_fixed(1))
    def correct(
        self,
        original_prompt: str,
        generated_sentences: list[str],
        incorrect_sentences: str,
        incorrect_facts: str,
    ) -> str:
        generated_sentences_str = "\n".join(
            [f"{i+1}. {sentence}" for i, sentence in enumerate(generated_sentences)]
        )
        result = self.chain.invoke(
            {
                "original_prompt": original_prompt,
                "generated_sentences": generated_sentences_str,
                "num_sentences": len(generated_sentences),
                "incorrect_sentences": incorrect_sentences,
                "incorrect_facts": incorrect_facts,
                "format": "\n".join([f"{i+1}. " for i in range(len(generated_sentences))]),
            }
        ).lower()
        assert isinstance(result, str)
        sentences = [s for s in result.split("\n") if s[0].isdigit()]
        assert len(sentences) == len(
            generated_sentences
        ), f'invalid result: "{result}" {len(sentences)=}'
        return result
