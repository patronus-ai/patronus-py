import datetime
import typing
from typing import Optional, Union

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
    evaluator_family: Optional[str]
    aliases: Optional[list[str]]


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
    profile_name: Optional[str] = None
    explain_strategy: str = "always"


class EvaluatedModelAttachment(pydantic.BaseModel):
    url: str
    media_type: str
    usage_type: str = "evaluated_model_input"


# See https://docs.patronus.ai/reference/evaluate_v1_evaluate_post for request field descriptions.
class EvaluateRequest(pydantic.BaseModel):
    # Currently we support calls with only one evaluator.
    # One of the reasons is that we support "smart" retires on failures,
    # and it wouldn't be possible with batch eval.
    evaluators: list[EvaluateEvaluator] = pydantic.Field(min_length=1, max_length=1)
    evaluated_model_system_prompt: Optional[str] = None
    evaluated_model_retrieved_context: Optional[list[str]] = None
    evaluated_model_input: Optional[str] = None
    evaluated_model_output: Optional[str] = None
    evaluated_model_gold_answer: Optional[str] = None
    evaluated_model_attachments: Optional[list[EvaluatedModelAttachment]] = None
    app: Optional[str] = None
    experiment_id: Optional[str] = None
    capture: str = "all"
    dataset_id: Optional[str] = None
    dataset_sample_id: Optional[int] = None
    tags: Optional[dict[str, str]] = None


class EvaluationResultAdditionalInfo(pydantic.BaseModel):
    positions: Optional[list]
    extra: Optional[dict]
    confidence_interval: Optional[dict]


class EvaluationResult(pydantic.BaseModel):
    id: str
    project_id: Optional[str]
    app: Optional[str]
    experiment_id: Optional[str]
    created_at: pydantic.AwareDatetime
    evaluator_id: str
    evaluated_model_system_prompt: Optional[str]
    evaluated_model_retrieved_context: Optional[list[str]]
    evaluated_model_input: Optional[str]
    evaluated_model_output: Optional[str]
    evaluated_model_gold_answer: Optional[str]
    pass_: Optional[bool] = pydantic.Field(alias="pass")
    score_raw: Optional[float]
    additional_info: EvaluationResultAdditionalInfo
    explanation: Optional[str]
    evaluation_duration: Optional[datetime.timedelta]
    explanation_duration: Optional[datetime.timedelta]
    evaluator_family: str
    evaluator_profile_public_id: str
    dataset_id: Optional[str]
    dataset_sample_id: Optional[int]
    tags: Optional[dict[str, str]]


class EvaluateResult(pydantic.BaseModel):
    evaluator_id: str
    profile_name: str
    status: str
    error_message: Optional[str]
    evaluation_result: Optional[EvaluationResult]


class EvaluateResponse(pydantic.BaseModel):
    results: list[EvaluateResult]


class ExportEvaluationResult(pydantic.BaseModel):
    experiment_id: str
    evaluator_id: str
    profile_name: Optional[str] = None
    evaluated_model_system_prompt: Optional[str] = None
    evaluated_model_retrieved_context: Optional[list[str]] = None
    evaluated_model_input: Optional[str] = None
    evaluated_model_output: Optional[str] = None
    evaluated_model_gold_answer: Optional[str] = None
    pass_: bool = pydantic.Field(alias="pass_", serialization_alias="pass")
    score_raw: Optional[float]
    evaluation_duration: Optional[datetime.timedelta] = None
    evaluated_model_name: Optional[str] = None
    evaluated_model_provider: Optional[str] = None
    evaluated_model_params: Optional[dict[str, Union[str, int, float]]] = None
    evaluated_model_selected_model: Optional[str] = None
    dataset_id: Optional[str] = None
    dataset_sample_id: Optional[int] = None
    tags: Optional[dict[str, str]] = None


class ExportEvaluationRequest(pydantic.BaseModel):
    evaluation_results: list[ExportEvaluationResult]


class ExportEvaluationResultPartial(pydantic.BaseModel):
    id: str
    app: Optional[str]
    created_at: pydantic.AwareDatetime
    evaluator_id: str


class ExportEvaluationResponse(pydantic.BaseModel):
    evaluation_results: list[ExportEvaluationResultPartial]


class ListProfilesRequest(pydantic.BaseModel):
    public_id: Optional[str] = None
    evaluator_family: Optional[str] = None
    evaluator_id: Optional[str] = None
    name: Optional[str] = None
    revision: Optional[str] = None
    get_last_revision: bool = False
    is_patronus_managed: Optional[bool] = None
    limit: int = 1000
    offset: int = 0


class EvaluatorProfile(pydantic.BaseModel):
    public_id: str
    evaluator_family: str
    name: str
    revision: int
    config: Optional[dict[str, typing.Any]]
    is_patronus_managed: bool
    created_at: datetime.datetime
    description: Optional[str]


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
    evaluated_model_system_prompt: Optional[str] = None
    evaluated_model_retrieved_context: Optional[list[str]] = None
    evaluated_model_input: Optional[str] = None
    evaluated_model_output: Optional[str] = None
    evaluated_model_gold_answer: Optional[str] = None
    meta_evaluated_model_name: Optional[str] = None
    meta_evaluated_model_provider: Optional[str] = None
    meta_evaluated_model_selected_model: Optional[str] = None
    meta_evaluated_model_params: Optional[dict[str, Union[str, int, float]]] = None


class ListDatasetData(pydantic.BaseModel):
    data: list[DatasetDatum]
