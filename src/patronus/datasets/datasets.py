import inspect

import asyncio
import dataclasses
import json
import pathlib
import re
import typing
from typing import Optional, Union

import pandas as pd
import typing_extensions as te


class Attachment(typing.TypedDict, total=False):
    """
    Represent an attachment entry. Usually used in context of multimodal evaluation.
    """

    media_type: str
    url: str


class Fields(typing.TypedDict, total=False):
    """
    A TypedDict class representing fields for a structured data entity.

    Attributes:
        sid: An optional identifier for the system or session.
        system_prompt: An optional string representing the system
            prompt associated with the task.
        task_context: Optional contextual
            information for the task in the form of a string or a list of strings.
        task_attachments: Optional list of attachments associated with the task.
        task_input: An optional string representing the input data for the task. Usually a user input sent to an LLM.
        task_output: An optional string representing the output result of the task. Usually a response from an LLM.
        gold_answer: An optional string representing the correct or expected answer for evaluation purposes.
        task_metadata: Optional dictionary containing metadata associated with the task.
        tags: Optional dictionary holding additional key-value pair tags relevant to the task.
    """

    sid: te.NotRequired[Optional[str]]
    system_prompt: te.NotRequired[Optional[str]]
    task_context: te.NotRequired[Union[str, list[str], None]]
    task_attachments: te.NotRequired[Optional[list[Attachment]]]
    task_input: te.NotRequired[Optional[str]]
    task_output: te.NotRequired[Optional[str]]
    gold_answer: te.NotRequired[Optional[str]]
    task_metadata: te.NotRequired[Optional[dict[str, typing.Any]]]
    tags: te.NotRequired[Optional[dict[str, str]]]


@dataclasses.dataclass
class Row:
    """
    Represents a data row encapsulating access to properties in a pandas Series.

    Provides attribute-based access to underlying pandas Series data with properties
    that ensure compatibility with structured evaluators through consistent field naming
    and type handling.
    """

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
    """
    Represents a dataset.
    """

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
        """
        Creates an instance of the class by processing and sanitizing provided records
        and optionally associating them with a specific dataset ID.

        Args:
            records:
                A collection of records to initialize the instance. Each record can either
                be an instance of `Fields` or a dictionary containing corresponding data.
            dataset_id:
                An optional identifier for associating the data with a specific dataset.

        Returns:
            te.Self: A new instance of the class with the processed and sanitized data.
        """
        df = pd.DataFrame.from_records(records)
        df = cls.__sanitize_df(df, dataset_id)
        return cls(df=df, dataset_id=dataset_id)

    def to_csv(
        self, path_or_buf: Union[str, pathlib.Path, typing.IO[typing.AnyStr]], **kwargs: typing.Any
    ) -> Optional[str]:
        """
        Saves dataset to a CSV file.

        Args:
            path_or_buf: String path or file-like object where the CSV will be saved.
            **kwargs: Additional arguments passed to pandas.DataFrame.to_csv().

        Returns:
            String path if a path was specified and return_path is True, otherwise None.
        """
        return self.df.to_csv(path_or_buf, **kwargs)

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
    **kwargs: typing.Any,
) -> Dataset:
    """
    Reads a CSV file and converts it into a Dataset object. The CSV file is transformed
    into a structured dataset where each field maps to a specific aspect of the dataset
    schema provided via function arguments. You may specify custom field mappings as per
    your dataset structure, while additional keyword arguments are passed directly to the
    underlying 'pd.read_csv' function.

    Args:
        filename_or_buffer: Path to the CSV file or a file-like object containing the
            dataset to be read.
        dataset_id: Optional identifier for the dataset being read. Default is None.
        sid_field: Name of the column containing unique sample identifiers.
        system_prompt_field: Name of the column representing the system prompts.
        task_input_field: Name of the column containing the main input for the task.
        task_context_field: Name of the column describing the broader task context.
        task_attachments_field: Name of the column with supplementary attachments
            related to the task.
        task_output_field: Name of the column containing responses or outputs for the
            task.
        gold_answer_field: Name of the column detailing the expected or correct
            answer to the task.
        task_metadata_field: Name of the column storing metadata attributes
            associated with the task.
        tags_field: Name of the column containing tags or annotations related to each
            sample.
        **kwargs: Additional keyword arguments passed to 'pandas.read_csv' for fine-tuning
            the CSV parsing behavior, such as delimiters, encoding, etc.

    Returns:
        Dataset: The parsed dataset object containing structured data from the input
            CSV file.
    """
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
    **kwargs: typing.Any,
) -> Dataset:
    """
    Reads a JSONL (JSON Lines) file and transforms it into a Dataset object. This function
    parses the input data file or buffer in JSON Lines format into a structured format,
    extracting specified fields and additional metadata for usage in downstream tasks. The
    field mappings and additional keyword arguments can be customized to accommodate
    application-specific requirements.

    Args:
        filename_or_buffer: The path to the file or a file-like object containing the JSONL
            data to be read.
        dataset_id: An optional identifier for the dataset being read. Defaults to None.
        sid_field: The field name in the JSON lines representing the unique identifier for
            a sample. Defaults to "sid".
        system_prompt_field: The field name for the system prompt in the JSON lines file.
            Defaults to "system_prompt".
        task_input_field: The field name for the task input data in the JSON lines file.
            Defaults to "task_input".
        task_context_field: The field name for the task context data in the JSON lines file.
            Defaults to "task_context".
        task_attachments_field: The field name for any task attachments in the JSON lines
            file. Defaults to "task_attachments".
        task_output_field: The field name for task output data in the JSON lines file.
            Defaults to "task_output".
        gold_answer_field: The field name for the gold (ground truth) answer in the JSON
            lines file. Defaults to "gold_answer".
        task_metadata_field: The field name for metadata associated with the task in the
            JSON lines file. Defaults to "task_metadata".
        tags_field: The field name for tags in the parsed JSON lines file. Defaults to
            "tags".
        **kwargs: Additional keyword arguments to be passed to `pd.read_json` for
            customization. The parameter "lines" will be forcibly set to True if not
            provided.

    Returns:
        Dataset: A Dataset object containing the parsed and structured data.

    """
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
    filename_or_buffer: Union[str, pathlib.Path, typing.IO[typing.AnyStr]],
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
    **kwargs: typing.Any,
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
    if tags_field in df.columns:
        df["tags"] = df[tags_field]

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
    """
    Encapsulates asynchronous loading of a dataset.

    This class provides a mechanism to lazily load a dataset asynchronously only
    once, using a provided dataset loader function.
    """

    def __init__(self, loader: Union[typing.Awaitable[Dataset], typing.Callable[[], typing.Awaitable[Dataset]]]):
        self.__lock = asyncio.Lock()
        self.__loader = loader
        self.dataset: Optional[Dataset] = None

    async def load(self) -> Dataset:
        """
        Load dataset. Repeated calls will return already loaded dataset.
        """
        async with self.__lock:
            if self.dataset is not None:
                return self.dataset
            if inspect.iscoroutinefunction(self.__loader):
                self.dataset = await self.__loader()
            else:
                self.dataset = await self.__loader
            return self.dataset
