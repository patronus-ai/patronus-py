import asyncio
import dataclasses
import pathlib
import re
import typing
import json

import pandas as pd


class Fields(typing.TypedDict):
    sid: typing.NotRequired[int | None]
    evaluated_model_system_prompt: typing.NotRequired[str | None]
    evaluated_model_retrieved_context: typing.NotRequired[str | list[str] | None]
    evaluated_model_input: typing.NotRequired[str | None]
    evaluated_model_gold_answer: typing.NotRequired[str | None]
    evaluated_model_name: typing.NotRequired[str | None]
    evaluated_model_provider: typing.NotRequired[str | None]
    evaluated_model_selected_model: typing.NotRequired[str | None]
    evaluated_model_params: typing.NotRequired[dict[str, typing.Any] | None]


@dataclasses.dataclass
class Row:
    _row: pd.Series

    def __getattr__(self, name: str):
        return self._row[name]

    @property
    def row(self) -> pd.Series:
        return self._row

    @property
    def dataset_id(self) -> str | None:
        return self._row.get("dataset_id")

    @property
    def sid(self) -> int:
        return self._row.sid

    @property
    def evaluated_model_system_prompt(self) -> str | None:
        if "evaluated_model_system_prompt" in self._row.index:
            return self._row.evaluated_model_system_prompt
        return None

    @property
    def evaluated_model_retrieved_context(self) -> list[str] | None:
        ctx = None
        if "evaluated_model_retrieved_context" in self._row.index:
            ctx = self._row.evaluated_model_retrieved_context
        if ctx is None:
            return None
        if isinstance(ctx, str):
            return [ctx]
        assert isinstance(ctx, list), f"evaluated_model_retrieved_context is not a list, its: {type(ctx)}"
        return ctx

    @property
    def evaluated_model_input(self) -> str | None:
        if "evaluated_model_input" in self._row.index:
            return self._row.evaluated_model_input
        return None

    @property
    def evaluated_model_output(self) -> str | None:
        if "evaluated_model_output" in self._row.index:
            return self._row.evaluated_model_output
        return None

    @property
    def evaluated_model_gold_answer(self) -> str | None:
        if "evaluated_model_gold_answer" in self._row.index:
            return self._row.evaluated_model_gold_answer
        return None


@dataclasses.dataclass
class Dataset:
    dataset_id: str | None
    df: pd.DataFrame

    def iterrows(self) -> typing.Iterable[Row]:
        for i, row in self.df.iterrows():
            yield Row(row)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, dataset_id: str | None = None) -> typing.Self:
        df = cls.__sanitize_df(df, dataset_id)
        return cls(df=df, dataset_id=dataset_id)

    @classmethod
    def from_records(
        cls, records: typing.Iterable[Fields] | typing.Iterable[dict[str, typing.Any]], dataset_id: str | None = None
    ) -> typing.Self:
        df = pd.DataFrame.from_records(records)
        df = cls.__sanitize_df(df, dataset_id)
        return cls(df=df, dataset_id=dataset_id)

    @classmethod
    def __sanitize_df(cls, df: pd.DataFrame, dataset_id: str) -> pd.DataFrame:
        # Validate and backfill "sid"
        if "sid" not in df.columns:
            df["sid"] = range(1, len(df) + 1)

        sid_count = df["sid"].count()
        if sid_count != 0 and sid_count != len(df):
            raise ValueError("Either all entries must have 'sid' or none should have it.")

        if sid_count == 0:
            df["sid"] = range(1, len(df) + 1)
        else:
            if df["sid"].duplicated().any():
                raise ValueError("Duplicate 'sid' values found. Dataset cannot have duplicated 'sid's.")

        if not pd.api.types.is_integer_dtype(df["sid"]):
            try:
                df["sid"] = df["sid"].astype(int)
            except ValueError:
                raise ValueError("'sid' column contains non-integer values that cannot be converted to integers.")

        def normalize_context(value) -> list[str] | None:
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

        if "evaluated_model_system_prompt" in df.columns:
            df["evaluated_model_system_prompt"] = df["evaluated_model_system_prompt"].astype("string[python]")
            df["evaluated_model_system_prompt"] = df["evaluated_model_system_prompt"].replace({pd.NA: None})
        if "evaluated_model_retrieved_context" in df.columns:
            df["evaluated_model_retrieved_context"] = df["evaluated_model_retrieved_context"].apply(normalize_context)
            df["evaluated_model_retrieved_context"] = df["evaluated_model_retrieved_context"].astype("object")
        if "evaluated_model_input" in df.columns:
            df["evaluated_model_input"] = df["evaluated_model_input"].astype("string[python]")
            df["evaluated_model_input"] = df["evaluated_model_input"].replace({pd.NA: None})
        if "evaluated_model_output" in df.columns:
            df["evaluated_model_output"] = df["evaluated_model_output"].astype("string[python]")
            df["evaluated_model_output"] = df["evaluated_model_output"].replace({pd.NA: None})
        if "evaluated_model_gold_answer" in df.columns:
            df["evaluated_model_gold_answer"] = df["evaluated_model_gold_answer"].astype("string[python]")
            df["evaluated_model_gold_answer"] = df["evaluated_model_gold_answer"].replace({pd.NA: None})

        # Backfill "dataset_id"
        if dataset_id:
            if "dataset_id" not in df.columns:
                df["dataset_id"] = dataset_id
            else:
                df["dataset_id"] = df["dataset_id"].fillna(dataset_id)

        df = df.sort_values("sid")
        return df


def read_csv(
    filename_or_buffer: str | pathlib.Path | typing.IO[typing.AnyStr],
    *,
    dataset_id: str | None = None,
    sid_field: str = "sid",
    evaluated_model_system_prompt_field: str = "evaluated_model_system_prompt",
    evaluated_model_retrieved_context_field: str = "evaluated_model_retrieved_context",
    evaluated_model_input_field: str = "evaluated_model_input",
    evaluated_model_output_field: str = "evaluated_model_output",
    evaluated_model_gold_answer_field: str = "evaluated_model_gold_answer",
    **kwargs,
) -> Dataset:
    return _read_dataframe(
        pd.read_csv,
        filename_or_buffer,
        dataset_id=dataset_id,
        sid_field=sid_field,
        evaluated_model_system_prompt_field=evaluated_model_system_prompt_field,
        evaluated_model_retrieved_context_field=evaluated_model_retrieved_context_field,
        evaluated_model_input_field=evaluated_model_input_field,
        evaluated_model_output_field=evaluated_model_output_field,
        evaluated_model_gold_answer_field=evaluated_model_gold_answer_field,
        **kwargs,
    )


def read_jsonl(
    filename_or_buffer,
    *,
    dataset_id: str | None = None,
    sid_field: str = "sid",
    evaluated_model_system_prompt_field: str = "evaluated_model_system_prompt",
    evaluated_model_input_field: str = "evaluated_model_input",
    evaluated_model_retrieved_context_field: str = "evaluated_model_retrieved_context",
    evaluated_model_output_field: str = "evaluated_model_output",
    evaluated_model_gold_answer_field: str = "evaluated_model_gold_answer",
    **kwargs,
) -> Dataset:
    kwargs.setdefault("lines", True)
    return _read_dataframe(
        pd.read_json,
        filename_or_buffer,
        dataset_id=dataset_id,
        sid_field=sid_field,
        evaluated_model_system_prompt_field=evaluated_model_system_prompt_field,
        evaluated_model_retrieved_context_field=evaluated_model_retrieved_context_field,
        evaluated_model_input_field=evaluated_model_input_field,
        evaluated_model_output_field=evaluated_model_output_field,
        evaluated_model_gold_answer_field=evaluated_model_gold_answer_field,
        **kwargs,
    )


def _read_dataframe(
    reader_function,
    filename_or_buffer,
    *,
    dataset_id: str | None = None,
    sid_field: str = "sid",
    evaluated_model_system_prompt_field: str = "evaluated_model_system_prompt",
    evaluated_model_retrieved_context_field: str = "evaluated_model_retrieved_context",
    evaluated_model_input_field: str = "evaluated_model_input",
    evaluated_model_output_field: str = "evaluated_model_output",
    evaluated_model_gold_answer_field: str = "evaluated_model_gold_answer",
    **kwargs,
) -> Dataset:
    df = reader_function(filename_or_buffer, **kwargs)

    if sid_field in df.columns:
        df["sid"] = df[sid_field]
    if evaluated_model_system_prompt_field in df.columns:
        df["evaluated_model_system_prompt"] = df[evaluated_model_system_prompt_field]
    if evaluated_model_retrieved_context_field in df.columns:
        df["evaluated_model_retrieved_context"] = df[evaluated_model_retrieved_context_field]
    if evaluated_model_input_field in df.columns:
        df["evaluated_model_input"] = df[evaluated_model_input_field]
    if evaluated_model_output_field in df.columns:
        df["evaluated_model_output"] = df[evaluated_model_output_field]
    if evaluated_model_gold_answer_field in df.columns:
        df["evaluated_model_gold_answer"] = df[evaluated_model_gold_answer_field]

    dataset_id = _sanitize_dataset_id(dataset_id)
    return Dataset.from_dataframe(df, dataset_id=dataset_id)


def _sanitize_dataset_id(dataset_id: str) -> str | None:
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
        self.dataset: Dataset | None = None

    async def load(self) -> Dataset:
        async with self.__lock:
            if self.dataset is not None:
                return self.dataset
            self.dataset = await self.__loader
            return self.dataset
