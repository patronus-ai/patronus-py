__all__ = [
    "DatasetDatum",
    "Dataset",
    "read_csv",
    "read_jsonl",
]

import os.path
import re
import csv
import dataclasses
import json
import typing


class DatasetDatum(typing.TypedDict):
    sid: typing.NotRequired[int | None]
    evaluated_model_system_prompt: typing.NotRequired[str | None]
    evaluated_model_retrieved_context: typing.NotRequired[list[str] | None]
    evaluated_model_input: typing.NotRequired[str | None]
    evaluated_model_output: typing.NotRequired[str | None]
    evaluated_model_gold_answer: typing.NotRequired[str | None]
    evaluated_model_name: typing.NotRequired[str | None]
    evaluated_model_provider: typing.NotRequired[str | None]
    evaluated_model_selected_model: typing.NotRequired[str | None]
    evaluated_model_params: typing.NotRequired[dict[str, int | float | str] | None]


@dataclasses.dataclass
class Dataset:
    dataset_id: str | None
    data: list[DatasetDatum]


def read_csv(
    filename: str,
    *,
    delimiter: str = ",",
    dataset_id: str | None = None,
    sid_field: str = "sid",
    evaluated_model_system_prompt_field: str = "evaluated_model_system_prompt",
    evaluated_model_input_field: str = "evaluated_model_input",
    evaluated_model_retrieved_context_field: str = "evaluated_model_retrieved_context",
    evaluated_model_output_field: str = "evaluated_model_output",
    evaluated_model_gold_answer_field: str = "evaluated_model_gold_answer",
    evaluated_model_name_field: str = "evaluated_model_name",
    evaluated_model_provider_field: str = "evaluated_model_provider",
    evaluated_model_selected_model_field: str = "evaluated_model_selected_model",
    evaluated_model_params_field: str = "evaluated_model_params",
) -> Dataset:
    data: list[DatasetDatum] = []
    with open(filename, mode="r", newline="") as fd:
        rd = csv.DictReader(fd, delimiter=delimiter)

        for i, row in enumerate(rd, start=1):
            r_ctx = _parse_retrieved_content(row.get(evaluated_model_retrieved_context_field, None))
            m_params = _parse_model_params(row.get(evaluated_model_params_field, None))
            data.append(
                DatasetDatum(
                    sid=row.get(sid_field, i),
                    evaluated_model_system_prompt=row.get(evaluated_model_system_prompt_field, None),
                    evaluated_model_retrieved_context=r_ctx,
                    evaluated_model_input=row.get(evaluated_model_input_field, None),
                    evaluated_model_output=row.get(evaluated_model_output_field, None),
                    evaluated_model_gold_answer=row.get(evaluated_model_gold_answer_field, None),
                    evaluated_model_name=row.get(evaluated_model_name_field, None),
                    evaluated_model_provider=row.get(evaluated_model_provider_field, None),
                    evaluated_model_selected_model=row.get(evaluated_model_selected_model_field, None),
                    evaluated_model_params=m_params,
                )
            )

    if dataset_id is None:
        dataset_id = _extract_filename(filename)
    dataset_id = _sanitize_dataset_id(dataset_id)

    return Dataset(
        dataset_id=dataset_id,
        data=data,
    )


def read_jsonl(
    filename: str,
    *,
    dataset_id: str | None = None,
    sid_field: str = "sid",
    evaluated_model_system_prompt_field: str = "evaluated_model_system_prompt",
    evaluated_model_input_field: str = "evaluated_model_input",
    evaluated_model_retrieved_context_field: str = "evaluated_model_retrieved_context",
    evaluated_model_output_field: str = "evaluated_model_output",
    evaluated_model_gold_answer_field: str = "evaluated_model_gold_answer",
    evaluated_model_name_field: str = "evaluated_model_name",
    evaluated_model_provider_field: str = "evaluated_model_provider",
    evaluated_model_selected_model_field: str = "evaluated_model_selected_model",
    evaluated_model_params_field: str = "evaluated_model_params",
) -> Dataset:
    data: list[DatasetDatum] = []
    with open(filename, mode="r") as fd:
        for i, line in enumerate(fd, start=1):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Parsing {filename!r}: JSON parsing failed on line {i!r}: {exc}")

            # TODO it would be good to validate that most of the fields are what we expect which is - str.
            data.append(
                DatasetDatum(
                    sid=entry.get(sid_field, i),
                    evaluated_model_system_prompt=entry.get(evaluated_model_system_prompt_field, None),
                    # TODO validated input
                    evaluated_model_retrieved_context=entry.get(evaluated_model_retrieved_context_field, None),
                    evaluated_model_input=entry.get(evaluated_model_input_field, None),
                    evaluated_model_output=entry.get(evaluated_model_output_field, None),
                    evaluated_model_gold_answer=entry.get(evaluated_model_gold_answer_field, None),
                    evaluated_model_name=entry.get(evaluated_model_name_field, None),
                    evaluated_model_provider=entry.get(evaluated_model_provider_field, None),
                    evaluated_model_selected_model=entry.get(evaluated_model_selected_model_field, None),
                    # TODO validated input
                    evaluated_model_params=entry.get(evaluated_model_params_field, None),
                )
            )

    if dataset_id is None:
        dataset_id = _extract_filename(filename)
    dataset_id = _sanitize_dataset_id(dataset_id)

    return Dataset(
        dataset_id=dataset_id,
        data=data,
    )


def _parse_retrieved_content(value: str | None) -> list[str] | None:
    if not value:
        return None
    try:
        data = json.loads(value)
        if isinstance(data, list) and all(isinstance(d, str) for d in data):
            return data
    except json.JSONDecodeError:
        ...

    # If we failed to parse retrieved context as an array of string,
    # then just return it as plain string
    return [value]


def _parse_model_params(value: str) -> dict[str, int | float | str] | None:
    if not value:
        return None
    try:
        data = json.loads(value)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None

    for key in data.keys():
        if not isinstance(data[key], (int, float, str)):
            del data[key]

    return data


def _sanitize_dataset_id(dataset_id: str) -> str | None:
    dataset_id = re.sub(r"[^a-zA-Z0-9\-_]", "-", dataset_id.strip())
    if not dataset_id:
        return None
    return dataset_id


def _extract_filename(filepath: str) -> str:
    filename = os.path.basename(filepath)
    f, _ = os.path.splitext(filename)
    return f
