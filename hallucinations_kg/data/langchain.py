from typing import Any

from json_repair import repair_json
from langchain.schema import BaseMessage
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.outputs import Generation
from langchain_core.runnables import RunnableLambda
from loguru import logger


def wrapper_repair_json(input: BaseMessage) -> BaseMessage:
    assert isinstance(input.content, str)
    new = repair_json(input.content, skip_json_loads=True)
    assert isinstance(new, str)
    input.content = new
    return input


def get_repair_json_runnable() -> RunnableLambda:
    return RunnableLambda(wrapper_repair_json)


class StrictPydanticOutputParser(PydanticOutputParser):
    def parse_result(self, result: list[Generation], *, partial: bool = False) -> Any:
        new_result = super().parse_result(result, partial=partial)
        if new_result is None:
            logger.error("Parsing result is None.")
            raise OutputParserException("Result is None")
        return new_result
