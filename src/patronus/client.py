import importlib.metadata
import logging
import typing
from typing import Optional, Union

import httpx

from . import api, api_types
from .config import config
from .datasets import Dataset, DatasetLoader
from .evaluators import Evaluator
from .evaluators_remote import RemoteEvaluator
from .tasks import Task

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
            raise ValueError("Provide 'api_key' argument or set PATRONUS_API_KEY environment variable.")

        if api_client is None:
            # TODO allow passing http client as an argument
            http_client = httpx.AsyncClient(timeout=timeout)
            http_sync_client = httpx.Client(timeout=timeout)

            api_client = api.API(
                version=importlib.metadata.version("patronus"),
                http=http_client,
                http_sync=http_sync_client,
            )

        api_client.set_target(base_url, api_key)
        self.api = api_client

    def evaluate(
        self,
        evaluator: str,
        criteria: Optional[str] = None,
        *,
        evaluated_model_system_prompt: Optional[str] = None,
        evaluated_model_retrieved_context: Optional[Union[list[str], str]] = None,
        evaluated_model_input: Optional[str] = None,
        evaluated_model_output: Optional[str] = None,
        evaluated_model_gold_answer: Optional[str] = None,
        evaluated_model_attachments: Optional[list[dict[str, typing.Any]]] = None,
        tags: Optional[dict[str, str]] = None,
        app: Optional[str] = None,
        experiment_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
        dataset_sample_id: Optional[int] = None,
        explain_strategy: typing.Literal["never", "on-fail", "on-success", "always"] = "always",
        capture: typing.Literal["all", "fails-only", "none"] = "all",
    ) -> api_types.EvaluationResult:
        """
        Synchronously evaluates a sample using specified remote evaluators.

        This is a convenience method primarily intended for quick testing and exploration.
        For production use cases or when you need features like automatic retries, use:
        - Client.remote_evaluator() to create a reusable evaluator instance with retry capabilities,
        - Client.api.evaluate() for batch async evaluation,
        - Client.api.evaluate_one() for single async evaluation with simplified error handling.

        Example:
        ```python
        from patronus import Client

        client = Client()
        result = client.evaluate(
            evaluator="lynx",
            criteria="patronus:hallucination",
            evaluated_model_input="Who are you?",
            evaluated_model_output="My name is Barry.",
            evaluated_model_retrieved_context="My name is John.",
        )
        ```
        """
        return self.api.evaluate_one_sync(
            api_types.EvaluateRequest(
                evaluators=[
                    api_types.EvaluateEvaluator(
                        evaluator=evaluator,
                        criteria=criteria,
                        explain_strategy=explain_strategy,
                    )
                ],
                evaluated_model_system_prompt=evaluated_model_system_prompt,
                evaluated_model_retrieved_context=evaluated_model_retrieved_context,
                evaluated_model_input=evaluated_model_input,
                evaluated_model_output=evaluated_model_output,
                evaluated_model_gold_answer=evaluated_model_gold_answer,
                evaluated_model_attachments=evaluated_model_attachments,
                tags=tags,
                app=app,
                experiment_id=experiment_id,
                dataset_id=dataset_id,
                dataset_sample_id=dataset_sample_id,
                capture=capture,
            )
        )

    def experiment(
        self,
        project_name: str,
        *,
        dataset=None,  # TODO type hint
        task: Optional[Task] = None,
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
        criteria: Optional[str] = None,
        *,
        explain_strategy: typing.Literal["never", "on-fail", "on-success", "always"] = "always",
        criteria_config: Optional[dict[str, typing.Any]] = None,
        allow_update: bool = False,
        max_attempts: int = 3,
    ) -> RemoteEvaluator:
        return RemoteEvaluator(
            evaluator_id_or_alias=evaluator_id_or_alias,
            criteria=criteria,
            explain_strategy=explain_strategy,
            criteria_config=criteria_config,
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
