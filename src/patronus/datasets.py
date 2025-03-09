import asyncio
import dataclasses
import json
import pathlib
import re
import typing
from typing import Optional, Union

import pandas as pd
import typing_extensions as te


class Fields(typing.TypedDict, total=False):
    sid: te.NotRequired[Optional[str]]
    system_prompt: te.NotRequired[Optional[str]]
    task_context: te.NotRequired[Union[str, list[str], None]]
    task_attachments: te.NotRequired[Optional[list[typing.Any]]]
    task_input: te.NotRequired[Optional[str]]
    task_output: te.NotRequired[Optional[str]]
    gold_answer: te.NotRequired[Optional[str]]
    task_metadata: te.NotRequired[Optional[dict[str, typing.Any]]]
    tags: te.NotRequired[Optional[dict[str, str]]]


@dataclasses.dataclass
class Row:
    _row: pd.Series

    def __getattr__(self, name: str):
        return self._row[name]

    @property
    def row(self) -> pd.Series:
        return self._row

    @property
    def dataset_id(self) -> Optional[str]:
        return self._row.get("dataset_id")

    @property
    def sid(self) -> str:
        return self._row.sid

    @property
    def system_prompt(self) -> Optional[str]:
        if "system_prompt" in self._row.index:
            return self._row.system_prompt
        return None

    @property
    def task_context(self) -> Optional[list[str]]:
        ctx = None
        if "task_context" in self._row.index:
            ctx = self._row.task_context
        if ctx is None:
            return None
        if isinstance(ctx, str):
            return [ctx]
        assert isinstance(ctx, list), f"task_context is not a list, its: {type(ctx)}"
        return ctx

    @property
    def task_attachments(self) -> Optional[list[typing.Any]]:
        attachments = None
        if "task_attachments" in self._row.index:
            attachments = self._row.task_attachments
        if attachments is None:
            return None
        return attachments

    @property
    def task_input(self) -> Optional[str]:
        if "task_input" in self._row.index:
            return self._row.task_input
        return None

    @property
    def task_output(self) -> Optional[str]:
        if "task_output" in self._row.index:
            return self._row.task_output
        return None

    @property
    def gold_answer(self) -> Optional[str]:
        if "gold_answer" in self._row.index:
            return self._row.gold_answer
        return None

    @property
    def task_metadata(self) -> Optional[dict[str, typing.Any]]:
        if "task_metadata" in self._row.index:
            return self._row.task_metadata
        return None

    @property
    def tags(self) -> Optional[dict[str, str]]:
        if "tags" in self._row.index:
            return self._row.tags
        return None


@dataclasses.dataclass
class Dataset:
    dataset_id: Optional[str]
    df: pd.DataFrame

    def iterrows(self) -> typing.Iterable[Row]:
        for i, row in self.df.iterrows():
            yield Row(row)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, dataset_id: Optional[str] = None) -> te.Self:
        df = cls.__sanitize_df(df, dataset_id)
        return cls(df=df, dataset_id=dataset_id)

    @classmethod
    def from_records(
        cls,
        records: Union[typing.Iterable[Fields], typing.Iterable[dict[str, typing.Any]]],
        dataset_id: Optional[str] = None,
    ) -> te.Self:
        df = pd.DataFrame.from_records(records)
        df = cls.__sanitize_df(df, dataset_id)
        return cls(df=df, dataset_id=dataset_id)

    @classmethod
    def __sanitize_df(cls, df: pd.DataFrame, dataset_id: str) -> pd.DataFrame:
        # Validate and backfill "sid"
        if "sid" not in df.columns:
            df["sid"] = range(1, len(df) + 1)

        sid_count = df["sid"].count()
        if sid_count == 0:
            df["sid"] = range(1, len(df) + 1)

        if not pd.api.types.is_string_dtype(df["sid"]):
            try:
                df["sid"] = df["sid"].astype(str)
            except ValueError:
                raise ValueError("'sid' column contains non-integer values that cannot be converted to integers.")

        def normalize_context(value) -> Optional[list[str]]:
            if value is None:
                return None

            if isinstance(value, list):
                return [str(v) for v in value if v]

            if pd.isna(value) or value == "" or value == "nan":
                return None

            if isinstance(value, str):
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, list):
                        return [str(v) for v in parsed if v]
                    else:
                        return [str(parsed)]
                except json.JSONDecodeError:
                    return [value]

            return [str(value)]

        def _assert_attachment(value: dict):
            assert isinstance(
                value.get("url"), str
            ), "parsing 'task_attachments': missing or invalid type (expected str) of 'url' field"
            assert isinstance(
                value.get("media_type"), str
            ), "parsing 'task_attachments': missing or invalid type (expected str) of 'media_type' field"
            usage_type = value.get("usage_type")
            assert (
                isinstance(usage_type, str) or usage_type is None
            ), "parsing 'task_attachments': invalid type (expected str) of 'usage_type' field"

        def _normalize_attachment(value) -> dict[str, typing.Any]:
            if isinstance(value, dict):
                _assert_attachment(value)
                return value

        def normalize_attachments(value) -> Optional[list[dict[str, typing.Any]]]:
            if value is None:
                return None

            if isinstance(value, list):
                return [_normalize_attachment(v) for v in value]

            if isinstance(value, dict):
                _assert_attachment(value)
                return [value]

            if isinstance(value, str):
                try:
                    return normalize_attachments(json.loads(value))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"parsing 'task_attachments': {exc}")

            raise ValueError("parsing 'task_attachments': unexpected value type")

        def normalize_metadata(value) -> Optional[dict[str, typing.Any]]:
            if value is None:
                return None
            if isinstance(value, dict):
                return value
            raise ValueError("parsing 'task_metadata': unexpected value type: expected dict or None")

        def normalize_tags(value) -> Optional[dict[str, str]]:
            if value is None:
                return None
            if isinstance(value, str):
                try:
                    return normalize_tags(json.loads(value))
                except json.JSONDecodeError as exc:
                    raise ValueError("parsing 'tags': unexpected value type") from exc
            if isinstance(value, dict):
                return value
            raise ValueError("parsing 'task_tags': unexpected value type: expected dict or None")

        if "system_prompt" in df.columns:
            df["system_prompt"] = df["system_prompt"].astype("string[python]")
            df["system_prompt"] = df["system_prompt"].replace({pd.NA: None})
        if "task_context" in df.columns:
            df["task_context"] = df["task_context"].apply(normalize_context)
            df["task_context"] = df["task_context"].astype("object")
        if "task_attachments" in df.columns:
            df["task_attachments"] = df["task_attachments"].apply(normalize_attachments)
            df["task_attachments"] = df["task_attachments"].astype("object")
        if "task_input" in df.columns:
            df["task_input"] = df["task_input"].astype("string[python]")
            df["task_input"] = df["task_input"].replace({pd.NA: None})
        if "task_output" in df.columns:
            df["task_output"] = df["task_output"].astype("string[python]")
            df["task_output"] = df["task_output"].replace({pd.NA: None})
        if "gold_answer" in df.columns:
            df["gold_answer"] = df["gold_answer"].astype("string[python]")
            df["gold_answer"] = df["gold_answer"].replace({pd.NA: None})
        if "task_metadata" in df.columns:
            df["task_metadata"] = df["task_metadata"].apply(normalize_metadata)
            df["task_metadata"] = df["task_metadata"].astype("object")
        if "tags" in df.columns:
            df["tags"] = df["tags"].apply(normalize_tags)
            df["tags"] = df["tags"].astype("object")

        # Backfill "dataset_id"
        if dataset_id:
            if "dataset_id" not in df.columns:
                df["dataset_id"] = dataset_id
            else:
                df["dataset_id"] = df["dataset_id"].fillna(dataset_id)

        df = df.sort_values("sid")
        return df


def read_csv(
    filename_or_buffer: Union[str, pathlib.Path, typing.IO[typing.AnyStr]],
    *,
    dataset_id: Optional[str] = None,
    sid_field: str = "sid",
    system_prompt_field: str = "system_prompt",
    task_input_field: str = "task_input",
    task_context_field: str = "task_context",
    task_attachments_field: str = "task_attachments",
    task_output_field: str = "task_output",
    gold_answer_field: str = "gold_answer",
    task_metadata_field: str = "task_metadata",
    tags_field: str = "tags",
    **kwargs,
) -> Dataset:
    return _read_dataframe(
        pd.read_csv,
        filename_or_buffer,
        dataset_id=dataset_id,
        sid_field=sid_field,
        system_prompt_field=system_prompt_field,
        task_context_field=task_context_field,
        task_attachments_field=task_attachments_field,
        task_input_field=task_input_field,
        task_output_field=task_output_field,
        gold_answer_field=gold_answer_field,
        task_metadata_field=task_metadata_field,
        tags_field=tags_field,
        **kwargs,
    )


def read_jsonl(
    filename_or_buffer,
    *,
    dataset_id: Optional[str] = None,
    sid_field: str = "sid",
    system_prompt_field: str = "system_prompt",
    task_input_field: str = "task_input",
    task_context_field: str = "task_context",
    task_attachments_field: str = "task_attachments",
    task_output_field: str = "task_output",
    gold_answer_field: str = "gold_answer",
    task_metadata_field: str = "task_metadata",
    tags_field: str = "tags",
    **kwargs,
) -> Dataset:
    kwargs.setdefault("lines", True)
    return _read_dataframe(
        pd.read_json,
        filename_or_buffer,
        dataset_id=dataset_id,
        sid_field=sid_field,
        system_prompt_field=system_prompt_field,
        task_context_field=task_context_field,
        task_attachments_field=task_attachments_field,
        task_input_field=task_input_field,
        task_output_field=task_output_field,
        gold_answer_field=gold_answer_field,
        task_metadata_field=task_metadata_field,
        tags_field=tags_field,
        **kwargs,
    )


def _read_dataframe(
    reader_function,
    filename_or_buffer,
    *,
    dataset_id: Optional[str] = None,
    sid_field: str = "sid",
    system_prompt_field: str = "system_prompt",
    task_context_field: str = "task_context",
    task_attachments_field: str = "task_attachments",
    task_input_field: str = "task_input",
    task_output_field: str = "task_output",
    gold_answer_field: str = "gold_answer",
    task_metadata_field: str = "task_metadata",
    tags_field: str = "tags",
    **kwargs,
) -> Dataset:
    df = reader_function(filename_or_buffer, **kwargs)

    if sid_field in df.columns:
        df["sid"] = df[sid_field]
    if system_prompt_field in df.columns:
        df["system_prompt"] = df[system_prompt_field]
    if task_context_field in df.columns:
        df["task_context"] = df[task_context_field]
    if task_attachments_field in df.columns:
        df["task_attachments"] = df[task_metadata_field]
    if task_input_field in df.columns:
        df["task_input"] = df[task_input_field]
    if task_output_field in df.columns:
        df["task_output"] = df[task_output_field]
    if gold_answer_field in df.columns:
        df["gold_answer"] = df[gold_answer_field]
    if task_metadata_field in df.columns:
        df["task_metadata"] = df[task_metadata_field]

    dataset_id = _sanitize_dataset_id(dataset_id)
    return Dataset.from_dataframe(df, dataset_id=dataset_id)


def _sanitize_dataset_id(dataset_id: str) -> Optional[str]:
    if not dataset_id:
        return None
    dataset_id = re.sub(r"[^a-zA-Z0-9\-_]", "-", dataset_id.strip())
    if not dataset_id:
        return None
    return dataset_id


class DatasetLoader:
    def __init__(self, loader: typing.Awaitable[Dataset]):
        self.__lock = asyncio.Lock()
        self.__loader = loader
        self.dataset: Optional[Dataset] = None

    async def load(self) -> Dataset:
        async with self.__lock:
            if self.dataset is not None:
                return self.dataset
            self.dataset = await self.__loader
            return self.dataset
