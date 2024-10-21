import asyncio
import logging
import typing

from . import api_types
from . import evaluators
from . import api
from .retry import retry


log = logging.getLogger(__name__)


class RemoteEvaluatorError(evaluators.EvaluatorError):
    def __init__(self, evaluator: str, profile_name: str, status: str, error_message: str):
        super().__init__(
            f"{evaluator!r} with profile {profile_name!r} "
            f"returned unexpected status {status!r} with error message {error_message!r}"
        )


class RemoteEvaluator(evaluators.Evaluator):
    remote_capture = True

    def __init__(
        self,
        evaluator_id_or_alias: str,
        profile_name: str,
        *,
        explain_strategy: typing.Literal["never", "on-fail", "on-success", "always"] = "always",
        profile_config: dict[str, typing.Any] | None = None,
        allow_update: bool = False,
        # Maximum number of attempts in case when evaluation throws an exception.
        max_attempts: int = 3,
        api_: api.API,
    ):
        self.__lock = asyncio.Lock()
        self.__initialized = False

        self.name = evaluator_id_or_alias
        self.evaluator = evaluator_id_or_alias

        self.profile_name = profile_name

        self.explain_strategy = explain_strategy
        self.profile_config = profile_config
        self.allow_update = allow_update
        self.max_attempts = max_attempts
        self.api = api_

        super().__init__(evaluators.EVALUATION_ARGS)

    def set_evaluator_ref(self, evaluator: str):
        self.name = evaluator
        self.evaluator = evaluator

    async def load(self) -> typing.Self:
        async with self.__lock:
            if self.__initialized:
                return

            evs = await self.api.list_evaluators()

            ev: api_types.Evaluator | None = None
            for e in evs:
                if e.id == self.evaluator:
                    ev = e
                for alias in e.aliases:
                    if alias == self.evaluator:
                        ev = e

            if ev is None:
                raise ValueError(f"Evaluator {self.evaluator!r} not found")

            if self.profile_config:
                await self._init_from_config(ev)
            else:
                await self._init_existing(ev=ev)

            self.__initialized = True
        return self

    async def _init_from_config(self, ev: api_types.Evaluator):
        if not self.profile_name:
            raise ValueError("profile_name is required when specifying profile_config")
        if self.profile_name.startswith("system:"):
            raise ValueError(
                f"Cannot use profile_config with system profiles. Provided profile was {self.profile_name!r}"
            )

        profiles = (
            await self.api.list_profiles(
                api_types.ListProfilesRequest(
                    evaluator_family=ev.evaluator_family,
                    name=self.profile_name,
                    get_last_revision=True,
                )
            )
        ).evaluator_profiles

        if not profiles:
            log.info(
                f"No evaluator profile {self.profile_name!r} for evaluator {ev.evaluator_family!r} found. Creating one..."
            )
            profile = (
                await self.api.create_profile(
                    api_types.CreateProfileRequest(
                        evaluator_family=ev.evaluator_family, name=self.profile_name, config=self.profile_config
                    )
                )
            ).evaluator_profile
            log.info(f"Evaluator profile {self.profile_name} created for evaluator family {ev.evaluator_family}.")
        elif len(profiles) > 1:
            raise Exception(
                f"Unexpected number of profiles retrieved for "
                f"evaluator {self.evaluator!r} and profile name {self.profile_name!r}"
            )
        else:
            profile = profiles[0]

        # Check if user provided profile config is subset of existing config
        # This checks only one level of the config, but we don't support profiles with nested
        # structure at this point so, it's alright.
        is_subset = {**profile.config, **self.profile_config} == profile.config

        if not is_subset and not self.allow_update:
            raise ValueError(
                "Provided 'profile_config' differs from existing profile. "
                "Please set 'allow_update=True' if you wish to update the profile. "
                "Updating profiles can be unsafe if they're used in production system or by other people."
            )

        if not is_subset:
            log.info("Existing profile config differs from the provided config. Adding revision to the profile...")
            profile_resp = await self.api.add_evaluator_profile_revision(
                profile.public_id,
                api_types.AddEvaluatorProfileRevisionRequest(
                    config={**profile.config, **self.profile_config},
                ),
            )
            log.info(f"Revision added to evaluator profile {profile_resp.evaluator_profile.name}.")

        self.set_evaluator_ref(ev.id)

    async def _init_existing(self, ev: api_types.Evaluator):
        profiles = await self.api.list_profiles(
            api_types.ListProfilesRequest(
                evaluator_family=ev.evaluator_family,
                name=self.profile_name,
                get_last_revision=True,
            )
        )
        if len(profiles.evaluator_profiles) == 0:
            raise ValueError(f"Profile for evaluator {self.evaluator!r} given name {self.profile_name!r} not found")
        if len(profiles.evaluator_profiles) > 1:
            raise ValueError(f"More than 1 profile found for evaluator {self.evaluator!r}")

        self.profile_name = profiles.evaluator_profiles[0].name
        self.set_evaluator_ref(ev.id)

    async def evaluate(
        self,
        *,
        evaluated_model_system_prompt: str | None = None,
        evaluated_model_retrieved_context: list[str] | None = None,
        evaluated_model_input: str | None = None,
        evaluated_model_output: str | None = None,
        evaluated_model_gold_answer: str | None = None,
        app: str | None = None,
        experiment_id: str | None = None,
        tags: dict[str, str] | None = None,
        dataset_id: str | None = None,
        dataset_sample_id: int | None = None,
        **kwargs,
    ) -> api_types.EvaluationResult:
        # Make sure that evaluator is loaded
        await self.load()

        @retry(max_attempts=self.max_attempts)
        async def call():
            return await self.api.evaluate(
                api_types.EvaluateRequest(
                    evaluators=[
                        api_types.EvaluateEvaluator(
                            evaluator=self.evaluator,
                            profile_name=self.profile_name,
                            explain_strategy=self.explain_strategy,
                        )
                    ],
                    evaluated_model_system_prompt=evaluated_model_system_prompt,
                    evaluated_model_retrieved_context=evaluated_model_retrieved_context,
                    evaluated_model_input=evaluated_model_input,
                    evaluated_model_output=evaluated_model_output,
                    evaluated_model_gold_answer=evaluated_model_gold_answer,
                    app=app,
                    experiment_id=experiment_id,
                    capture="all",
                    dataset_id=dataset_id,
                    dataset_sample_id=dataset_sample_id,
                    tags=tags,
                )
            )

        response = await call()
        data = response.results[0]
        if data.status != "success":
            raise RemoteEvaluatorError(self.evaluator, self.profile_name, data.status, data.error_message)

        return data.evaluation_result