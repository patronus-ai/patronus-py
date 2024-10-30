import asyncio
import inspect
import logging
import typing
from typing import Optional

import typing_extensions as te

from . import api, api_types, evaluators, types
from .datasets import Row
from .retry import retry

log = logging.getLogger(__name__)


class RemoteEvaluatorError(evaluators.EvaluatorError):
    def __init__(self, evaluator: str, criteria: str, status: str, error_message: str):
        super().__init__(
            f"{evaluator!r} with criteria {criteria!r} "
            f"returned unexpected status {status!r} with error message {error_message!r}"
        )


class EvaluateCall(typing.Protocol):
    def __call__(
        self,
        evaluated_model_system_prompt: Optional[str] = None,
        evaluated_model_retrieved_context: Optional[list[str]] = None,
        evaluated_model_input: Optional[str] = None,
        evaluated_model_output: Optional[str] = None,
        evaluated_model_gold_answer: Optional[str] = None,
        evaluated_model_attachments: Optional[list[dict[str, typing.Any]]] = None,
    ) -> typing.Awaitable[api_types.EvaluationResult]: ...


class EvaluateWrapperFunction(typing.Protocol):
    def __call__(
        self,
        call: EvaluateCall,
        *,
        row: Row,
        task_result: typing.Optional[types.TaskResult] = None,
        evaluated_model_system_prompt: Optional[str] = None,
        evaluated_model_retrieved_context: Optional[list[str]] = None,
        evaluated_model_input: Optional[str] = None,
        evaluated_model_output: Optional[str] = None,
        evaluated_model_gold_answer: Optional[str] = None,
        evaluated_model_attachments: Optional[list[dict[str, typing.Any]]] = None,
        parent: types.EvalParent = None,
        **kwargs,
    ) -> typing.Awaitable[api_types.EvaluateResponse]: ...


class RemoteEvaluator(evaluators.Evaluator):
    remote_capture = True

    def __init__(
        self,
        evaluator_id_or_alias: str,
        criteria: str,
        *,
        explain_strategy: typing.Literal["never", "on-fail", "on-success", "always"] = "always",
        criteria_config: Optional[dict[str, typing.Any]] = None,
        allow_update: bool = False,
        # Maximum number of attempts in case when evaluation throws an exception.
        max_attempts: int = 3,
        api_: api.API,
        override_func: Optional[EvaluateWrapperFunction] = None,
    ):
        self.__lock = asyncio.Lock()
        self.__initialized = False

        self.name = evaluator_id_or_alias
        self.evaluator = evaluator_id_or_alias

        self.criteria = criteria

        self.explain_strategy = explain_strategy
        self.criteria_config = criteria_config
        self.allow_update = allow_update
        self.max_attempts = max_attempts
        self.api = api_

        self.override_func = override_func

        super().__init__(evaluators.EVALUATION_ARGS)

    def set_evaluator_ref(self, evaluator: str):
        self.name = evaluator
        self.evaluator = evaluator

    async def load(self) -> te.Self:
        async with self.__lock:
            if self.__initialized:
                return

            evs = await self.api.list_evaluators()

            ev: Optional[api_types.Evaluator] = None
            for e in evs:
                if e.id == self.evaluator:
                    ev = e
                for alias in e.aliases:
                    if alias == self.evaluator:
                        ev = e

            if ev is None:
                raise ValueError(f"Evaluator {self.evaluator!r} not found")

            if self.criteria_config:
                await self._init_from_config(ev)
            else:
                await self._init_existing(ev=ev)

            self.__initialized = True
        return self

    async def _init_from_config(self, ev: api_types.Evaluator):
        if not self.criteria:
            raise ValueError("criteria is required when specifying criteria_config")
        if self.criteria.startswith("system:"):
            raise ValueError(
                f"Cannot use criteria_config with patronus criteria. Provided criteria was {self.criteria!r}"
            )

        criteria_list = (
            await self.api.list_criteria(
                api_types.ListCriteriaRequest(
                    evaluator_family=ev.evaluator_family,
                    name=self.criteria,
                    get_last_revision=True,
                )
            )
        ).evaluator_criteria

        if not criteria_list:
            log.info(
                f"No evaluator criteria {self.criteria!r} for evaluator {ev.evaluator_family!r} found. Creating one..."
            )
            criteria = (
                await self.api.create_criteria(
                    api_types.CreateCriteriaRequest(
                        evaluator_family=ev.evaluator_family,
                        name=self.criteria,
                        config=self.criteria_config,
                    )
                )
            ).evaluator_criteria
            log.info(f"Evaluator criteria {self.criteria} created for evaluator family {ev.evaluator_family}.")
        elif len(criteria_list) > 1:
            raise Exception(
                f"Unexpected number of criteria retrieved for "
                f"evaluator {self.evaluator!r} and criteria name {self.criteria!r}"
            )
        else:
            criteria = criteria_list[0]

        # Check if user provided criteria config is subset of existing config
        # This checks only one level of the config, but we don't support criteria with nested
        # structure at this point so, it's alright.
        is_subset = {**criteria.config, **self.criteria_config} == criteria.config

        if not is_subset and not self.allow_update:
            raise ValueError(
                "Provided 'criteria_config' differs from existing criteria. "
                "Please set 'allow_update=True' if you wish to update the criteria. "
                "Updating criteria can be unsafe if they're used in production system or by other people."
            )

        if not is_subset:
            log.info("Existing criteria config differs from the provided config. Adding revision to the criteria...")
            criteria_resp = await self.api.add_evaluator_criteria_revision(
                criteria.public_id,
                api_types.AddEvaluatorCriteriaRevisionRequest(
                    config={**criteria.config, **self.criteria_config},
                ),
            )
            log.info(f"Revision added to evaluator criteria {criteria_resp.evaluator_criteria.name}.")

        self.set_evaluator_ref(ev.id)

    async def _init_existing(self, ev: api_types.Evaluator):
        criteria_list = await self.api.list_criteria(
            api_types.ListCriteriaRequest(
                evaluator_family=ev.evaluator_family,
                name=self.criteria,
                get_last_revision=True,
            )
        )
        if len(criteria_list.evaluator_criteria) == 0:
            raise ValueError(f"Criteria for evaluator {self.evaluator!r} given name {self.criteria!r} not found")
        if len(criteria_list.evaluator_criteria) > 1:
            raise ValueError(f"More than 1 criteria found for evaluator {self.evaluator!r}")

        self.criteria = criteria_list.evaluator_criteria[0].name
        self.set_evaluator_ref(ev.id)

    async def evaluate(
        self,
        *,
        evaluated_model_system_prompt: Optional[str] = None,
        evaluated_model_retrieved_context: Optional[list[str]] = None,
        evaluated_model_input: Optional[str] = None,
        evaluated_model_output: Optional[str] = None,
        evaluated_model_gold_answer: Optional[str] = None,
        evaluated_model_attachments: Optional[list[dict[str, typing.Any]]] = None,
        tags: Optional[dict[str, str]] = None,
        dataset_id: Optional[str] = None,
        dataset_sample_id: Optional[int] = None,
        app: Optional[str] = None,
        # experiment_id is generally used by the framework.
        experiment_id: Optional[str] = None,
        # row is framework specific field. It's not necessary when using evaluator outside of framework.
        row: Optional[Row] = None,
        # task_result is framework specific field. It's not necessary when using evaluator outside of framework.
        task_result: Optional[types.TaskResult] = None,
        # parent is framework specific field. It's not necessary when using evaluator outside of framework.
        parent: types.EvalParent = None,
        **kwargs,
    ) -> api_types.EvaluationResult:
        # Make sure that evaluator is loaded
        await self.load()

        @retry(max_attempts=self.max_attempts)
        async def call(
            evaluated_model_system_prompt: Optional[str] = None,
            evaluated_model_retrieved_context: Optional[list[str]] = None,
            evaluated_model_input: Optional[str] = None,
            evaluated_model_output: Optional[str] = None,
            evaluated_model_gold_answer: Optional[str] = None,
            evaluated_model_attachments: Optional[list[dict[str, typing.Any]]] = None,
        ) -> api_types.EvaluationResult:
            return await self.api.evaluate_one(
                api_types.EvaluateRequest(
                    evaluators=[
                        api_types.EvaluateEvaluator(
                            evaluator=self.evaluator,
                            criteria=self.criteria,
                            explain_strategy=self.explain_strategy,
                        )
                    ],
                    evaluated_model_system_prompt=evaluated_model_system_prompt,
                    evaluated_model_retrieved_context=evaluated_model_retrieved_context,
                    evaluated_model_input=evaluated_model_input,
                    evaluated_model_output=evaluated_model_output,
                    evaluated_model_gold_answer=evaluated_model_gold_answer,
                    evaluated_model_attachments=evaluated_model_attachments,
                    app=app,
                    experiment_id=experiment_id,
                    capture="all",
                    dataset_id=dataset_id,
                    dataset_sample_id=dataset_sample_id,
                    tags=tags,
                )
            )

        if self.override_func is not None:
            try:
                response = self.override_func(
                    call,
                    row=row,
                    task_result=task_result,
                    evaluated_model_system_prompt=evaluated_model_system_prompt,
                    evaluated_model_retrieved_context=evaluated_model_retrieved_context,
                    evaluated_model_input=evaluated_model_input,
                    evaluated_model_output=evaluated_model_output,
                    evaluated_model_gold_answer=evaluated_model_gold_answer,
                    evaluated_model_attachments=evaluated_model_attachments,
                    parent=parent,
                )
            except TypeError as exc:
                if "got an unexpected keyword argument" in "".join(exc.args):
                    raise TypeError(f"{exc}. Did you forgot to add '**kwargs' to the function parameters?")
                raise exc
            if inspect.iscoroutine(response):
                response = await response

            if response is None:
                return None
        else:
            response = await call(
                evaluated_model_system_prompt=evaluated_model_system_prompt,
                evaluated_model_retrieved_context=evaluated_model_retrieved_context,
                evaluated_model_input=evaluated_model_input,
                evaluated_model_output=evaluated_model_output,
                evaluated_model_gold_answer=evaluated_model_gold_answer,
                evaluated_model_attachments=evaluated_model_attachments,
            )

        return response

    def wrap(self, fn: EvaluateWrapperFunction) -> "RemoteEvaluator":
        return RemoteEvaluator(
            evaluator_id_or_alias=self.name,
            criteria=self.criteria,
            explain_strategy=self.explain_strategy,
            criteria_config=self.criteria_config,
            allow_update=self.allow_update,
            max_attempts=self.max_attempts,
            api_=self.api,
            override_func=fn,
        )
