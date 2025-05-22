from typing import Optional

from .datasets import Dataset, DatasetLoader
from patronus.context import get_api_client_deprecated


class DatasetNotFoundError(Exception):
    """Raised when a dataset with the specified ID or name is not found"""

    pass


class RemoteDatasetLoader(DatasetLoader):
    """
    A loader for datasets stored remotely on the Patronus platform.

    This class provides functionality to asynchronously load a dataset from
    the remote API by its name or identifier, handling the fetch operation lazily
    and ensuring it's only performed once. You can specify either the dataset name or ID,
    but not both.
    """

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

    async def _load(self) -> Dataset:
        api = get_api_client_deprecated()

        # If we're loading by name, first find the dataset ID by listing datasets
        dataset_id = self._dataset_id
        if self._dataset_name:
            datasets = await api.list_datasets()
            matching_datasets = [d for d in datasets if d.name == self._dataset_name]

            if not matching_datasets:
                raise DatasetNotFoundError(f"No dataset found with name '{self._dataset_name}'")
            if len(matching_datasets) > 1:
                raise ValueError(
                    f"Multiple datasets found with name '{self._dataset_name}'. Please use a dataset ID instead."
                )

            dataset_id = matching_datasets[0].id

        # Make sure we have a valid dataset ID
        if not dataset_id:
            raise ValueError("Unable to determine dataset ID")

        resp = await api.list_dataset_data(dataset_id)
        data = resp.model_dump()["data"]

        if not data:
            raise DatasetNotFoundError(f"Dataset with ID '{dataset_id}' not found or contains no data")

        records = [
            {
                "sid": datum.get("sid"),
                "system_prompt": datum.get("evaluated_model_system_prompt"),
                "task_context": datum.get("evaluated_model_retrieved_context"),
                "task_attachments": None,
                "task_input": datum.get("evaluated_model_input"),
                "task_output": datum.get("evaluated_model_output"),
                "gold_answer": datum.get("evaluated_model_gold_answer"),
                "task_metadata": None,
                "tags": None,
            }
            for datum in data
        ]
        return Dataset.from_records(records, dataset_id=dataset_id)
