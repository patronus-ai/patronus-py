from .datasets import Dataset, DatasetLoader
from patronus.context import get_api_client


class RemoteDataset(Dataset):
    """
    Represents a dataset that is fetched from a remote source via the Patronus API.
    """

    @classmethod
    async def fetch(cls, dataset_id: str) -> "RemoteDataset":
        """
        Fetches a dataset from the Patronus API by its ID and creates a RemoteDataset instance.

        This method retrieves dataset records from the API, transforms them into the expected
        format, and constructs a new RemoteDataset instance containing the retrieved data.

        Arguments:
            dataset_id: A string identifier for the dataset to be fetched from the API.

        Returns:
            RemoteDataset: A new RemoteDataset instance containing the fetched data.
        """
        api = get_api_client()
        resp = await api.list_dataset_data(dataset_id)
        data = resp.model_dump()["data"]
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
        return cls.from_records(records, dataset_id=dataset_id)


class RemoteDatasetLoader(DatasetLoader):
    """
    A loader for datasets stored remotely on the Patronus platform.

    This class provides functionality to asynchronously load a dataset from
    the remote API by its identifier, handling the fetch operation lazily
    and ensuring it's only performed once.
    """

    def __init__(self, dataset_id: str):
        """
        Initializes a new RemoteDatasetLoader instance.

        Arguments:
            dataset_id: A string identifier for the remote dataset to be loaded.
        """
        self._dataset_id = dataset_id
        super().__init__(self._load)

    async def _load(self) -> RemoteDataset:
        return await RemoteDataset.fetch(self._dataset_id)
