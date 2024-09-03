import logging
import typing

import httpx

from ._config import config
from ._dataset import Dataset, DatasetDatum
from ._evaluators import Evaluator
from ._evaluators_remote import RemoteEvaluator
from ._tasks import Task, nop_task
from . import _api as api

log = logging.getLogger(__name__)


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
        *,
        data: list[DatasetDatum] | typing.Callable[[...], list[DatasetDatum]] | typing.Awaitable[list[DatasetDatum]],
        task: Task = nop_task,
        evaluators: list[Evaluator],
        tags: dict[str, str] | None = None,
        experiment_name: str = "",
        max_concurrency: int = 10,
        **kwargs,
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
            max_concurrency=max_concurrency,
            **kwargs,
        )

    async def remote_evaluator(
        self,
        # ID or an alias of an evaluator.
        evaluator: str,
        # profile_name is not necessary for evaluators that not requires them, like "toxicity".
        profile_name: str | None = None,
        *,
        profile_config: dict[str, typing.Any] | None = None,
        allow_update: bool = False,
        explain_strategy: typing.Literal["never", "on-fail", "on-success", "always"] = "always",
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

        if profile_config:
            return await self._remote_evaluator_from_config(
                evaluator=evaluator,
                profile_name=profile_name,
                ev=ev,
                profile_config=profile_config,
                allow_update=allow_update,
                explain_strategy=explain_strategy,
            )
        else:
            return await self._remote_evaluator(
                evaluator=evaluator,
                profile_name=profile_name,
                ev=ev,
                explain_strategy=explain_strategy,
            )

    async def _remote_evaluator_from_config(
        self,
        *,
        evaluator: str,
        profile_name: str,
        ev: api.Evaluator,
        profile_config: dict[str, typing.Any],
        allow_update: bool,
        explain_strategy: typing.Literal["never", "on-fail", "on-success", "always"],
    ) -> RemoteEvaluator:
        if not profile_name:
            raise ValueError("profile_name is required when specifying profile_config")
        if profile_name.startswith("system:"):
            raise ValueError(f"Cannot use profile_config with system profiles. Provided profile was {profile_name!r}")

        profiles = (
            await self.api.list_profiles(
                api.ListProfilesRequest(
                    evaluator_family=ev.evaluator_family,
                    name=profile_name,
                    get_last_revision=True,
                )
            )
        ).evaluator_profiles

        if not profiles:
            log.info(
                f"No evaluator profile {profile_name!r} for evaluator {ev.evaluator_family!r} found. Creating one..."
            )
            profile = await self._create_profile_from_config(ev.evaluator_family, profile_name, profile_config)
            log.info(f"Evaluator profile {profile_name} created for evaluator family {ev.evaluator_family}.")
        elif len(profiles) > 1:
            raise Exception(
                f"Unexpected number of profiles retrieved for "
                f"evaluator {evaluator!r} and profile name {profile_name!r}"
            )
        else:
            profile = profiles[0]

        # Check if user provided profile config is subset of existing config
        # This checks only one level of the config, but we don't support profiles with nested
        # structure at this point so, it's alright.
        is_subset = {**profile.config, **profile_config} == profile.config

        if not is_subset and not allow_update:
            raise ValueError(
                "Provided 'profile_config' differs from existing profile. "
                "Please set 'allow_update=True' if you wish to update the profile. "
                "Updating profiles can be unsafe if they're used in production system or by other people."
            )

        if not is_subset:
            log.info("Existing profile config differs from the provided config. Adding revision to the profile...")
            profile = (
                await self.api.add_evaluator_profile_revision(
                    profile.public_id,
                    api.AddEvaluatorProfileRevisionRequest(
                        config={**profile.config, **profile_config},
                    ),
                )
            ).evaluator_profile
            log.info(f"Revision added to evaluator profile {profile_name}.")

        return RemoteEvaluator(
            evaluator=ev.id,
            profile_name=profile.name,
            explain_strategy=explain_strategy,
            api_=self.api,
        )

    async def _create_profile_from_config(
        self,
        evaluator_family: str,
        profile_name: str,
        profile_config: dict[str, typing.Any],
    ) -> api.EvaluatorProfile:
        resp = await self.api.create_profile(
            api.CreateProfileRequest(
                evaluator_family=evaluator_family,
                name=profile_name,
                config=profile_config,
            )
        )
        return resp.evaluator_profile

    async def _remote_evaluator(
        self,
        *,
        evaluator: str,
        profile_name: str | None,
        ev: api.Evaluator,
        explain_strategy: typing.Literal["never", "on-fail", "on-success", "always"],
    ) -> RemoteEvaluator:
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
            explain_strategy=explain_strategy,
            api_=self.api,
        )

    async def remote_dataset(self, dataset_id: str) -> Dataset:
        resp = await self.api.list_dataset_data(dataset_id)
        return Dataset(dataset_id=dataset_id, data=resp.model_dump()["data"])
