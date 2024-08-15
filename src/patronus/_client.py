import os
import typing

import httpx

from ._async_utils import run_until_complete
from ._evaluators import Evaluator
from ._evaluators_remote import RemoteEvaluator
from ._tasks import Task
from ._api import API


class Client:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.patronus.ai",
        api: API | None = None,
        # TODO Allow passing more types for the timeout: float, Timeout, None, NotSet
        timeout: float = 300,
    ):
        if api_key is None:
            api_key = os.environ.get("PATRONUSAI_API_KEY")

        if not api_key:
            raise ValueError("Provide 'api_key' argument or set PATRONUSAI_API_KEY environment variable.")

        if api is None:
            # TODO allow passing http client as an argument
            http_client = httpx.AsyncClient(timeout=timeout)

            # TODO use package version
            api = API(version="0.0.1", http=http_client)
        api.set_target(base_url, api_key)
        self.api = api

    def experiment(
        self,
        name: str,
        data: list[dict],
        task: Task,
        evaluators: list[Evaluator],
        tags: dict[str, str] | None = None,
        display_hist: bool = False,
    ):
        from ._experiment import experiment as ex

        ex(
            self,
            name=name,
            data=data,
            task=task,
            evaluators=evaluators,
            tags=tags,
            display_hist=display_hist,
        )

    def remote_evaluator(
        self,
        # TODO family should not be necessary but we need to get aliases from the API
        family: str,
        # ID or an alias of an evaluator.
        evaluator: str,
        profile_name: str,
        # TODO
        # config: dict[str, typing.Any] | None = None,
        # allow_upsert: bool = False,
    ) -> RemoteEvaluator:
        # TODO this should be awaited
        profile = run_until_complete(self.api.get_profile(family, profile_name))
        return RemoteEvaluator(
            evaluator=evaluator,
            profile_name=profile.name,
            api_=self.api,
        )

    def remote_dataset(self, dataset_id: str) -> list[dict[str, typing.Any]]:
        data = run_until_complete(self.api.list_dataset_data(dataset_id))
        return data.model_dump()["data"]

    def get_project_by_name(self, project_name: str):
        # TODO
        pass
