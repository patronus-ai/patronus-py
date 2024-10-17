import datetime
import typing

import pydantic


class Account(pydantic.BaseModel):
    id: str
    name: str


class WhoAmIAPIKey(pydantic.BaseModel):
    id: str
    account: Account


class WhoAmICaller(pydantic.BaseModel):
    api_key: WhoAmIAPIKey


class WhoAmIResponse(pydantic.BaseModel):
    caller: WhoAmICaller


class Evaluator(pydantic.BaseModel):
    id: str
    name: str
    evaluator_family: str | None
    aliases: list[str] | None


class ListEvaluatorsResponse(pydantic.BaseModel):
    evaluators: list[Evaluator]


class Project(pydantic.BaseModel):
    id: str
    name: str


class CreateProjectRequest(pydantic.BaseModel):
    name: str


class GetProjectResponse(pydantic.BaseModel):
    project: Project


class Experiment(pydantic.BaseModel):
    project_id: str
    id: str
    name: str


class CreateExperimentRequest(pydantic.BaseModel):
    project_id: str
    name: str


class CreateExperimentResponse(pydantic.BaseModel):
    experiment: Experiment


class GetExperimentResponse(pydantic.BaseModel):
    experiment: Experiment


class EvaluateEvaluator(pydantic.BaseModel):
    evaluator: str
    profile_name: str | None = None
    explain_strategy: str = "always"


# See https://docs.patronus.ai/reference/evaluate_v1_evaluate_post for request field descriptions.
class EvaluateRequest(pydantic.BaseModel):
    # Currently we support calls with only one evaluator.
    # One of the reasons is that we support "smart" retires on failures,
    # and it wouldn't be possible with batch eval.
    evaluators: list[EvaluateEvaluator] = pydantic.Field(min_length=1, max_length=1)
    evaluated_model_system_prompt: str | None = None
    evaluated_model_retrieved_context: list[str] | None = None
    evaluated_model_input: str | None = None
    evaluated_model_output: str | None = None
    evaluated_model_gold_answer: str | None = None
    app: str | None = None
    experiment_id: str | None = None
    capture: str = "all"
    dataset_id: str | None = None
    dataset_sample_id: int | None = None
    tags: dict[str, str] | None = None


class EvaluationResultAdditionalInfo(pydantic.BaseModel):
    positions: list | None
    extra: dict | None
    confidence_interval: dict | None


class EvaluationResult(pydantic.BaseModel):
    id: str
    project_id: str | None
    app: str | None
    experiment_id: str | None
    created_at: pydantic.AwareDatetime
    evaluator_id: str
    evaluated_model_system_prompt: str | None
    evaluated_model_retrieved_context: list[str] | None
    evaluated_model_input: str | None
    evaluated_model_output: str | None
    evaluated_model_gold_answer: str | None
    pass_: bool | None = pydantic.Field(alias="pass")
    score_raw: float | None
    additional_info: EvaluationResultAdditionalInfo
    explanation: str | None
    evaluation_duration: datetime.timedelta | None
    explanation_duration: datetime.timedelta | None
    evaluator_family: str
    evaluator_profile_public_id: str
    dataset_id: str | None
    dataset_sample_id: int | None
    tags: dict[str, str] | None


class EvaluateResult(pydantic.BaseModel):
    evaluator_id: str
    profile_name: str
    status: str
    error_message: str | None
    evaluation_result: EvaluationResult | None


class EvaluateResponse(pydantic.BaseModel):
    results: list[EvaluateResult]


class ExportEvaluationResult(pydantic.BaseModel):
    experiment_id: str
    evaluator_id: str
    profile_name: str | None = None
    evaluated_model_system_prompt: str | None = None
    evaluated_model_retrieved_context: list[str] | None = None
    evaluated_model_input: str | None = None
    evaluated_model_output: str | None = None
    evaluated_model_gold_answer: str | None = None
    pass_: bool = pydantic.Field(alias="pass_", serialization_alias="pass")
    score_raw: float | None
    evaluation_duration: datetime.timedelta | None = None
    evaluated_model_name: str | None = None
    evaluated_model_provider: str | None = None
    evaluated_model_params: dict[str, str | int | float] | None = None
    evaluated_model_selected_model: str | None = None
    dataset_id: str | None = None
    dataset_sample_id: int | None = None
    tags: dict[str, str] | None = None


class ExportEvaluationRequest(pydantic.BaseModel):
    evaluation_results: list[ExportEvaluationResult]


class ExportEvaluationResultPartial(pydantic.BaseModel):
    id: str
    app: str | None
    created_at: pydantic.AwareDatetime
    evaluator_id: str


class ExportEvaluationResponse(pydantic.BaseModel):
    evaluation_results: list[ExportEvaluationResultPartial]


class ListProfilesRequest(pydantic.BaseModel):
    public_id: str | None = None
    evaluator_family: str | None = None
    evaluator_id: str | None = None
    name: str | None = None
    revision: str | None = None
    get_last_revision: bool = False
    is_patronus_managed: bool | None = None
    limit: int = 1000
    offset: int = 0


class EvaluatorProfile(pydantic.BaseModel):
    public_id: str
    evaluator_family: str
    name: str
    revision: int
    config: dict[str, typing.Any] | None
    is_patronus_managed: bool
    created_at: datetime.datetime
    description: str | None


class CreateProfileRequest(pydantic.BaseModel):
    evaluator_family: str
    name: str
    config: dict[str, typing.Any]


class CreateProfileResponse(pydantic.BaseModel):
    evaluator_profile: EvaluatorProfile


class AddEvaluatorProfileRevisionRequest(pydantic.BaseModel):
    config: dict[str, typing.Any]


class AddEvaluatorProfileRevisionResponse(pydantic.BaseModel):
    evaluator_profile: EvaluatorProfile


class ListProfilesResponse(pydantic.BaseModel):
    evaluator_profiles: list[EvaluatorProfile]


class DatasetDatum(pydantic.BaseModel):
    dataset_id: str
    sid: int
    evaluated_model_system_prompt: str | None = None
    evaluated_model_retrieved_context: list[str] | None = None
    evaluated_model_input: str | None = None
    evaluated_model_output: str | None = None
    evaluated_model_gold_answer: str | None = None
    meta_evaluated_model_name: str | None = None
    meta_evaluated_model_provider: str | None = None
    meta_evaluated_model_selected_model: str | None = None
    meta_evaluated_model_params: dict[str, str | int | float] | None = None


class ListDatasetData(pydantic.BaseModel):
    data: list[DatasetDatum]
