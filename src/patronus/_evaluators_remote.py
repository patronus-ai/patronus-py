import typing

from . import _evaluators as evaluators
from . import _api as api


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
        evaluator: str,
        profile_name: str,
        explain_strategy: typing.Literal["never", "on-fail", "on-success", "always"],
        api_: api.API,
    ):
        self.name = evaluator
        self.evaluator = evaluator
        self.profile_name = profile_name
        self.explain_strategy = explain_strategy
        self.api = api_

        super().__init__(evaluators.EVALUATION_ARGS)

    async def evaluate(
        self,
        experiment_id: str,
        evaluated_model_system_prompt: str | None = None,
        evaluated_model_retrieved_context: list[str] | None = None,
        evaluated_model_input: str | None = None,
        evaluated_model_output: str | None = None,
        evaluated_model_gold_answer: str | None = None,
        dataset_id: str | None = None,
        dataset_sample_id: int | None = None,
        tags: dict[str, str] | None = None,
    ) -> evaluators.EvaluationResultT:
        # TODO error handling
        response = await self.api.evaluate(
            api.EvaluateRequest(
                evaluators=[
                    api.EvaluateEvaluator(
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
                experiment_id=experiment_id,
                capture="all",
                dataset_id=dataset_id,
                dataset_sample_id=dataset_sample_id,
                tags=tags,
            )
        )
        data = response.results[0]
        if data.status != "success":
            raise RemoteEvaluatorError(self.evaluator, self.profile_name, data.status, data.error_message)

        return data.evaluation_result
