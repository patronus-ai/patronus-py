# Datasets

## patronus.datasets

### datasets

#### Attachment

Bases: `TypedDict`

Represent an attachment entry. Usually used in context of multimodal evaluation.

#### Fields

Bases: `TypedDict`

A TypedDict class representing fields for a structured data entity.

Attributes:

| Name | Type | Description | | --- | --- | --- | | `sid` | `NotRequired[Optional[str]]` | An optional identifier for the system or session. | | `system_prompt` | `NotRequired[Optional[str]]` | An optional string representing the system prompt associated with the task. | | `task_context` | `NotRequired[Union[str, list[str], None]]` | Optional contextual information for the task in the form of a string or a list of strings. | | `task_attachments` | `NotRequired[Optional[list[Attachment]]]` | Optional list of attachments associated with the task. | | `task_input` | `NotRequired[Optional[str]]` | An optional string representing the input data for the task. Usually a user input sent to an LLM. | | `task_output` | `NotRequired[Optional[str]]` | An optional string representing the output result of the task. Usually a response from an LLM. | | `gold_answer` | `NotRequired[Optional[str]]` | An optional string representing the correct or expected answer for evaluation purposes. | | `task_metadata` | `NotRequired[Optional[dict[str, Any]]]` | Optional dictionary containing metadata associated with the task. | | `tags` | `NotRequired[Optional[dict[str, str]]]` | Optional dictionary holding additional key-value pair tags relevant to the task. |

#### Row

```python
Row(_row: Series)

```

Represents a data row encapsulating access to properties in a pandas Series.

Provides attribute-based access to underlying pandas Series data with properties that ensure compatibility with structured evaluators through consistent field naming and type handling.

#### Dataset

```python
Dataset(dataset_id: Optional[str], df: DataFrame)

```

Represents a dataset.

##### from_records

```python
from_records(
    records: Union[
        Iterable[Fields], Iterable[dict[str, Any]]
    ],
    dataset_id: Optional[str] = None,
) -> te.Self

```

Creates an instance of the class by processing and sanitizing provided records and optionally associating them with a specific dataset ID.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `records` | `Union[Iterable[Fields], Iterable[dict[str, Any]]]` | A collection of records to initialize the instance. Each record can either be an instance of Fields or a dictionary containing corresponding data. | *required* | | `dataset_id` | `Optional[str]` | An optional identifier for associating the data with a specific dataset. | `None` |

Returns:

| Type | Description | | --- | --- | | `Self` | te.Self: A new instance of the class with the processed and sanitized data. |

Source code in `src/patronus/datasets/datasets.py`

```python
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

```

##### to_csv

```python
to_csv(
    path_or_buf: Union[str, Path, IO[AnyStr]], **kwargs: Any
) -> Optional[str]

```

Saves dataset to a CSV file.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `path_or_buf` | `Union[str, Path, IO[AnyStr]]` | String path or file-like object where the CSV will be saved. | *required* | | `**kwargs` | `Any` | Additional arguments passed to pandas.DataFrame.to_csv(). | `{}` |

Returns:

| Type | Description | | --- | --- | | `Optional[str]` | String path if a path was specified and return_path is True, otherwise None. |

Source code in `src/patronus/datasets/datasets.py`

```python
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

```

#### DatasetLoader

```python
DatasetLoader(
    loader: Union[
        Awaitable[Dataset], Callable[[], Awaitable[Dataset]]
    ],
)

```

Encapsulates asynchronous loading of a dataset.

This class provides a mechanism to lazily load a dataset asynchronously only once, using a provided dataset loader function.

Source code in `src/patronus/datasets/datasets.py`

```python
def __init__(self, loader: Union[typing.Awaitable[Dataset], typing.Callable[[], typing.Awaitable[Dataset]]]):
    self.__lock = asyncio.Lock()
    self.__loader = loader
    self.dataset: Optional[Dataset] = None

```

##### load

```python
load() -> Dataset

```

Load dataset. Repeated calls will return already loaded dataset.

Source code in `src/patronus/datasets/datasets.py`

```python
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

```

#### read_csv

```python
read_csv(
    filename_or_buffer: Union[str, Path, IO[AnyStr]],
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
    **kwargs: Any,
) -> Dataset

```

Reads a CSV file and converts it into a Dataset object. The CSV file is transformed into a structured dataset where each field maps to a specific aspect of the dataset schema provided via function arguments. You may specify custom field mappings as per your dataset structure, while additional keyword arguments are passed directly to the underlying 'pd.read_csv' function.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `filename_or_buffer` | `Union[str, Path, IO[AnyStr]]` | Path to the CSV file or a file-like object containing the dataset to be read. | *required* | | `dataset_id` | `Optional[str]` | Optional identifier for the dataset being read. Default is None. | `None` | | `sid_field` | `str` | Name of the column containing unique sample identifiers. | `'sid'` | | `system_prompt_field` | `str` | Name of the column representing the system prompts. | `'system_prompt'` | | `task_input_field` | `str` | Name of the column containing the main input for the task. | `'task_input'` | | `task_context_field` | `str` | Name of the column describing the broader task context. | `'task_context'` | | `task_attachments_field` | `str` | Name of the column with supplementary attachments related to the task. | `'task_attachments'` | | `task_output_field` | `str` | Name of the column containing responses or outputs for the task. | `'task_output'` | | `gold_answer_field` | `str` | Name of the column detailing the expected or correct answer to the task. | `'gold_answer'` | | `task_metadata_field` | `str` | Name of the column storing metadata attributes associated with the task. | `'task_metadata'` | | `tags_field` | `str` | Name of the column containing tags or annotations related to each sample. | `'tags'` | | `**kwargs` | `Any` | Additional keyword arguments passed to 'pandas.read_csv' for fine-tuning the CSV parsing behavior, such as delimiters, encoding, etc. | `{}` |

Returns:

| Name | Type | Description | | --- | --- | --- | | `Dataset` | `Dataset` | The parsed dataset object containing structured data from the input CSV file. |

Source code in `src/patronus/datasets/datasets.py`

```python
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

```

#### read_jsonl

```python
read_jsonl(
    filename_or_buffer: Union[str, Path, IO[AnyStr]],
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
    **kwargs: Any,
) -> Dataset

```

Reads a JSONL (JSON Lines) file and transforms it into a Dataset object. This function parses the input data file or buffer in JSON Lines format into a structured format, extracting specified fields and additional metadata for usage in downstream tasks. The field mappings and additional keyword arguments can be customized to accommodate application-specific requirements.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `filename_or_buffer` | `Union[str, Path, IO[AnyStr]]` | The path to the file or a file-like object containing the JSONL data to be read. | *required* | | `dataset_id` | `Optional[str]` | An optional identifier for the dataset being read. Defaults to None. | `None` | | `sid_field` | `str` | The field name in the JSON lines representing the unique identifier for a sample. Defaults to "sid". | `'sid'` | | `system_prompt_field` | `str` | The field name for the system prompt in the JSON lines file. Defaults to "system_prompt". | `'system_prompt'` | | `task_input_field` | `str` | The field name for the task input data in the JSON lines file. Defaults to "task_input". | `'task_input'` | | `task_context_field` | `str` | The field name for the task context data in the JSON lines file. Defaults to "task_context". | `'task_context'` | | `task_attachments_field` | `str` | The field name for any task attachments in the JSON lines file. Defaults to "task_attachments". | `'task_attachments'` | | `task_output_field` | `str` | The field name for task output data in the JSON lines file. Defaults to "task_output". | `'task_output'` | | `gold_answer_field` | `str` | The field name for the gold (ground truth) answer in the JSON lines file. Defaults to "gold_answer". | `'gold_answer'` | | `task_metadata_field` | `str` | The field name for metadata associated with the task in the JSON lines file. Defaults to "task_metadata". | `'task_metadata'` | | `tags_field` | `str` | The field name for tags in the parsed JSON lines file. Defaults to "tags". | `'tags'` | | `**kwargs` | `Any` | Additional keyword arguments to be passed to pd.read_json for customization. The parameter "lines" will be forcibly set to True if not provided. | `{}` |

Returns:

| Name | Type | Description | | --- | --- | --- | | `Dataset` | `Dataset` | A Dataset object containing the parsed and structured data. |

Source code in `src/patronus/datasets/datasets.py`

```python
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

```

### remote

#### DatasetNotFoundError

Bases: `Exception`

Raised when a dataset with the specified ID or name is not found

#### RemoteDatasetLoader

```python
RemoteDatasetLoader(
    by_name: Optional[str] = None,
    *,
    by_id: Optional[str] = None,
)

```

Bases: `DatasetLoader`

A loader for datasets stored remotely on the Patronus platform.

This class provides functionality to asynchronously load a dataset from the remote API by its name or identifier, handling the fetch operation lazily and ensuring it's only performed once. You can specify either the dataset name or ID, but not both.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `by_name` | `Optional[str]` | The name of the dataset to load. | `None` | | `by_id` | `Optional[str]` | The ID of the dataset to load. | `None` |

Source code in `src/patronus/datasets/remote.py`

```python
def __init__(self, by_name: Optional[str] = None, *, by_id: Optional[str] = None):
    """
    Initializes a new RemoteDatasetLoader instance.

    Args:
        by_name: The name of the dataset to load.
        by_id: The ID of the dataset to load.
    """
    if not (bool(by_name) ^ bool(by_id)):
        raise ValueError("Either by_name or by_id must be provided, but not both.")

    self._dataset_name = by_name
    self._dataset_id = by_id
    super().__init__(self._load)

```
