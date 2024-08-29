import httpx

from ._config import config
from ._dataset import Dataset
from ._evaluators import Evaluator
from ._evaluators_remote import RemoteEvaluator
from ._tasks import Task
from . import _api as api


class Client:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "",
        api_client: api.API | None = None,
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

            # TODO use package version
            api_client = api.API(version="0.0.1", http=http_client)
        api_client.set_target(base_url, api_key)
        self.api = api_client

    def experiment(
        self,
        project_name: str,
        data: list[dict],
        task: Task,
        evaluators: list[Evaluator],
        tags: dict[str, str] | None = None,
        experiment_name: str = "",
    ):
        from ._experiment import experiment as ex

        ex(
            self,
            project_name=project_name,
            data=data,
            task=task,
            evaluators=evaluators,
            tags=tags,
            experiment_name=experiment_name,
        )

    async def remote_evaluator(
        self,
        # ID or an alias of an evaluator.
        evaluator: str,
        # profile_name is not necessary for evaluators that not requires them, like "toxicity".
        profile_name: str | None = None,
    ) -> RemoteEvaluator:
        evaluators = await self.api.list_evaluators()

        ev: api.Evaluator | None = None
        for e in evaluators:
            if e.id == evaluator:
                ev = e
            for alias in e.aliases:
                if alias == evaluator:
                    ev = e

        if ev is None:
            raise ValueError(f"Evaluator {evaluator!r} not found")

        profiles = await self.api.list_profiles(
            api.ListProfilesRequest(
                evaluator_family=ev.evaluator_family,
                name=profile_name,
                get_last_revision=True,
            )
        )
        if len(profiles.evaluator_profiles) == 0:
            raise ValueError(f"Profile for evaluator {evaluator!r} given name {profile_name!r} not found")
        if len(profiles.evaluator_profiles) > 1:
            raise ValueError(f"More than 1 profile found for evaluator {evaluator!r}")

        profile = profiles.evaluator_profiles[0]

        return RemoteEvaluator(
            evaluator=ev.id,
            profile_name=profile.name,
            api_=self.api,
        )

    async def remote_dataset(self, dataset_id: str) -> Dataset:
        resp = await self.api.list_dataset_data(dataset_id)
        return Dataset(dataset_id=dataset_id, data=resp.model_dump()["data"])
