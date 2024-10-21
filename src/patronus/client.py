import logging
import typing
import importlib.metadata
from typing import Optional

import httpx

from .config import config
from .evaluators import Evaluator
from .evaluators_remote import RemoteEvaluator
from .tasks import Task, nop_task
from . import api
from .datasets import Dataset, DatasetLoader

log = logging.getLogger(__name__)


class Client:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "",
        api_client: Optional[api.API] = None,
        # TODO Allow passing more types for the timeout: float, Timeout, None, NotSet
        timeout: float = 300,
    ):
        api_key = api_key or config().api_key
        base_url = base_url or config().api_url

        if not api_key:
            raise ValueError("Provide 'api_key' argument or set PATRONUSAI_API_KEY environment variable.")

        if api_client is None:
            # TODO allow passing http client as an argument
            http_client = httpx.AsyncClient(timeout=timeout)

            api_client = api.API(version=importlib.metadata.version("patronus"), http=http_client)
        api_client.set_target(base_url, api_key)
        self.api = api_client

    def experiment(
        self,
        project_name: str,
        *,
        dataset=None,  # TODO type hint
        task: Task = nop_task,
        evaluators: Optional[list[Evaluator]] = None,
        chain: Optional[list[dict[str, typing.Any]]] = None,
        tags: Optional[dict[str, str]] = None,
        experiment_name: str = "",
        max_concurrency: int = 10,
        experiment_id: Optional[str] = None,
        **kwargs,
    ):
        from .experiment import experiment as ex

        return ex(
            self,
            project_name=project_name,
            dataset=dataset,
            task=task,
            evaluators=evaluators,
            chain=chain,
            tags=tags,
            experiment_name=experiment_name,
            max_concurrency=max_concurrency,
            experiment_id=experiment_id,
            **kwargs,
        )

    def remote_evaluator(
        self,
        evaluator_id_or_alias: str,
        profile_name: Optional[str] = None,
        *,
        explain_strategy: typing.Literal["never", "on-fail", "on-success", "always"] = "always",
        profile_config: Optional[dict[str, typing.Any]] = None,
        allow_update: bool = False,
        max_attempts: int = 3,
    ) -> RemoteEvaluator:
        return RemoteEvaluator(
            evaluator_id_or_alias=evaluator_id_or_alias,
            profile_name=profile_name,
            explain_strategy=explain_strategy,
            profile_config=profile_config,
            allow_update=allow_update,
            max_attempts=max_attempts,
            api_=self.api,
        )

    def remote_dataset(self, dataset_id: str) -> DatasetLoader:
        async def load_dataset():
            resp = await self.api.list_dataset_data(dataset_id)
            data = resp.model_dump()["data"]
            return Dataset.from_records(data, dataset_id=dataset_id)

        return DatasetLoader(load_dataset())
