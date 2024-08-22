import dataclasses
import typing


class DatasetDatum(typing.TypedDict):
    sid: typing.NotRequired[int | None]
    evaluated_model_system_prompt: typing.NotRequired[str | None]
    evaluated_model_retrieved_context: typing.NotRequired[list[str] | None]
    evaluated_model_input: typing.NotRequired[str | None]
    evaluated_model_output: typing.NotRequired[str | None]
    evaluated_model_gold_answer: typing.NotRequired[str | None]


@dataclasses.dataclass
class Dataset:
    dataset_id: str | None
    data: list[DatasetDatum]
