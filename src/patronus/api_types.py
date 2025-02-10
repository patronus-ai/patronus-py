import datetime
import pydantic
import re
import typing
import uuid
from typing import Optional, Union
from typing_extensions import Annotated


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
    tags: Optional[dict[str, str]] = None


class CreateExperimentRequest(pydantic.BaseModel):
    project_id: str
    name: str
    tags: dict[str, str] = pydantic.Field(default_factory=dict)


class CreateExperimentResponse(pydantic.BaseModel):
    experiment: Experiment


class GetExperimentResponse(pydantic.BaseModel):
    experiment: Experiment


class EvaluateEvaluator(pydantic.BaseModel):
    evaluator: str
    criteria: Optional[str] = None
    explain_strategy: str = "always"


class EvaluatedModelAttachment(pydantic.BaseModel):
    url: str
    media_type: str
    usage_type: Optional[str] = "evaluated_model_input"


# See https://docs.patronus.ai/reference/evaluate_v1_evaluate_post for request field descriptions.
class EvaluateRequest(pydantic.BaseModel):
    evaluators: list[EvaluateEvaluator] = pydantic.Field(min_length=1)
    evaluated_model_system_prompt: Optional[str] = None
    evaluated_model_retrieved_context: Optional[Union[list[str], str]] = None
    evaluated_model_input: Optional[str] = None
    evaluated_model_output: Optional[str] = None
    evaluated_model_gold_answer: Optional[str] = None
    evaluated_model_attachments: Optional[list[EvaluatedModelAttachment]] = None
    project_id: Optional[str] = None
    project_name: Optional[str] = None
    app: Optional[str] = None
    experiment_id: Optional[str] = None
    capture: str = "all"
    dataset_id: Optional[str] = None
    dataset_sample_id: Optional[int] = None
    tags: Optional[dict[str, str]] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    log_id: Optional[str] = None


class EvaluationResult(pydantic.BaseModel):
    id: Optional[str] = None
    project_id: Optional[str] = None
    app: Optional[str] = None
    experiment_id: Optional[str] = None
    created_at: Optional[pydantic.AwareDatetime] = None
    evaluator_id: str
    criteria: str
    evaluated_model_system_prompt: Optional[str] = None
    evaluated_model_retrieved_context: Optional[list[str]] = None
    evaluated_model_input: Optional[str] = None
    evaluated_model_output: Optional[str] = None
    evaluated_model_gold_answer: Optional[str] = None
    pass_: Optional[bool] = pydantic.Field(default=None, alias="pass")
    score_raw: Optional[float] = None
    text_output: Optional[str] = None
    additional_info: Optional[dict[str, typing.Any]] = None
    evaluation_metadata: Optional[dict] = None
    explanation: Optional[str] = None
    evaluation_duration: Optional[datetime.timedelta] = None
    explanation_duration: Optional[datetime.timedelta] = None
    evaluator_family: str
    evaluator_profile_public_id: str
    dataset_id: Optional[str] = None
    dataset_sample_id: Optional[int] = None
    tags: Optional[dict[str, str]] = None


class EvaluateResult(pydantic.BaseModel):
    evaluator_id: str
    criteria: str
    status: str
    error_message: Optional[str]
    evaluation_result: Optional[EvaluationResult]


class EvaluateResponse(pydantic.BaseModel):
    results: list[EvaluateResult]


def sanitize_evaluator_id(v: typing.Any, info: pydantic.ValidationInfo):
    if not isinstance(v, str):
        return v
    v = v.strip()
    return re.sub(r"[^a-zA-Z0-9\-_./]", "-", v)


class ExportEvaluationResult(pydantic.BaseModel):
    app: Optional[str] = None
    experiment_id: Optional[str] = None
    evaluator_id: Annotated[str, pydantic.BeforeValidator(sanitize_evaluator_id)]
    criteria: Optional[str] = None
    evaluated_model_system_prompt: Optional[str] = None
    evaluated_model_retrieved_context: Optional[list[str]] = None
    evaluated_model_input: Optional[str] = None
    evaluated_model_output: Optional[str] = None
    evaluated_model_gold_answer: Optional[str] = None
    evaluated_model_attachments: Optional[list[EvaluatedModelAttachment]] = None
    pass_: Optional[bool] = pydantic.Field(default=None, serialization_alias="pass")
    score_raw: Optional[float] = None
    text_output: Optional[str] = None
    explanation: Optional[str] = None
    evaluation_duration: Optional[datetime.timedelta] = None
    explanation_duration: Optional[datetime.timedelta] = None
    evaluation_metadata: Optional[dict[str, typing.Any]] = None
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


class ListCriteriaRequest(pydantic.BaseModel):
    public_id: Optional[str] = None
    evaluator_family: Optional[str] = None
    evaluator_id: Optional[str] = None
    name: Optional[str] = None
    revision: Optional[str] = None
    get_last_revision: bool = False
    is_patronus_managed: Optional[bool] = None
    limit: int = 1000
    offset: int = 0


class EvaluatorCriteria(pydantic.BaseModel):
    public_id: str
    evaluator_family: str
    name: str
    revision: int
    config: Optional[dict[str, typing.Any]]
    is_patronus_managed: bool
    created_at: datetime.datetime
    description: Optional[str]


class CreateCriteriaRequest(pydantic.BaseModel):
    evaluator_family: str
    name: str
    config: dict[str, typing.Any]


class CreateCriteriaResponse(pydantic.BaseModel):
    evaluator_criteria: EvaluatorCriteria


class AddEvaluatorCriteriaRevisionRequest(pydantic.BaseModel):
    config: dict[str, typing.Any]


class AddEvaluatorCriteriaRevisionResponse(pydantic.BaseModel):
    evaluator_criteria: EvaluatorCriteria


class ListCriteriaResponse(pydantic.BaseModel):
    evaluator_criteria: list[EvaluatorCriteria]


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


def sanitize_field(max_length: int, sub_pattern: str):
    def wrapper(value: str) -> str:
        if not value:
            return value
        value = value[:max_length]
        return re.sub(sub_pattern, "_", value).strip()

    return wrapper


class Evaluation(pydantic.BaseModel):
    id: int
    log_id: str
    created_at: Optional[datetime.datetime] = None

    project_id: Optional[str] = None
    app: Optional[str] = None
    experiment_id: Optional[int] = None

    evaluator_family: Optional[str] = None
    evaluator_id: Optional[str] = None
    criteria_id: Optional[str] = None
    criteria: Optional[str] = None
    explain_strategy: Optional[str] = None
    pass_: Optional[bool] = pydantic.Field(default=None, alias="pass")

    score: Optional[float] = None
    text_output: Optional[str] = None
    metadata: Optional[dict[str, typing.Any]] = None
    explanation: Optional[str] = None
    evaluation_duration: Optional[datetime.timedelta] = None
    explanation_duration: Optional[datetime.timedelta] = None
    usage: Optional[dict[str, typing.Any]] = None
    metric_name: Optional[str] = None
    metric_description: Optional[str] = None
    annotation_criteria_id: Optional[str] = None
    created_at: datetime.datetime
    evaluation_type: Optional[str] = None
    tags: Optional[dict[str, str]] = None
    dataset_id: Optional[str] = None
    dataset_sample_id: Optional[str] = None


class ClientEvaluation(pydantic.BaseModel):
    log_id: uuid.UUID
    project_id: Optional[str] = None
    project_name: Optional[str] = None
    app: typing.Annotated[
        Optional[str],
        pydantic.BeforeValidator(sanitize_field(50, r"[^a-zA-Z0-9_./ -]")),
    ]
    experiment_id: Optional[str] = None
    evaluator_id: str
    criteria: Optional[str] = None
    pass_: Optional[bool] = pydantic.Field(default=None, alias="pass")
    score: Optional[float] = None
    text_output: Optional[str] = None
    metadata: Optional[dict[str, typing.Any]] = None
    explanation: Optional[str] = None
    evaluation_duration: Optional[datetime.timedelta] = None
    explanation_duration: Optional[datetime.timedelta] = None
    metric_name: Optional[str] = None
    metric_description: Optional[str] = None
    dataset_id: Optional[str] = None
    dataset_sample_id: Optional[int] = None
    created_at: Optional[datetime.datetime] = None
    tags: Optional[dict[str, str]] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None


class GetEvaluationResponse(pydantic.BaseModel):
    evaluation: Evaluation


class BatchCreateEvaluationsRequest(pydantic.BaseModel):
    evaluations: list[ClientEvaluation] = pydantic.Field(
        min_length=1,
        max_length=1000,
    )


class BatchCreateEvaluationsResponse(pydantic.BaseModel):
    evaluations: list[Evaluation]
