from typing import Any, Self

import pydantic
from pydantic import BaseModel


class _Base(BaseModel):
    @classmethod
    def model_validate(  # type: ignore
        cls,
        obj: Any,
        *,
        strict: bool | None = None,
        from_attributes: bool | None = None,
        context: Any | None = None,
    ) -> Self:
        try:
            return super().model_validate(
                obj, strict=strict, from_attributes=from_attributes, context=context
            )
        except pydantic.ValidationError:
            if isinstance(obj, list) and len(obj) == 1:
                return super().model_validate(
                    obj[0], strict=strict, from_attributes=from_attributes, context=context
                )


class Entities(_Base):
    entities: list[str | int]


class Relations(_Base):
    relationship_types: list[str]
