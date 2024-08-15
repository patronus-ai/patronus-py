import datetime
import typing

import pydantic

from ._base_api import BaseAPIClient


class EvaluateEvaluator(pydantic.BaseModel):
    evaluator: str
    profile_name: str | None = None
    explain_strategy: str = "always"


class EvaluateRequest(pydantic.BaseModel):
    evaluators: list[EvaluateEvaluator]
    evaluated_model_system_prompt: str | None = None
    evaluated_model_retrieved_context: list[str] | None = None
    evaluated_model_input: str | None = None
    evaluated_model_output: str | None = None
    evaluated_model_gold_answer: str | None = None

    app: str
    capture: str = "all"
    tags: dict[str, str] | None = None


class EvaluationResultAdditionalInfo(pydantic.BaseModel):
    positions: list | None
    extra: dict | None
    confidence_interval: dict | None


class EvaluationResult(pydantic.BaseModel):
    id: str
    app: str
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
    tags: dict[str, str] | None


class EvaluateResult(pydantic.BaseModel):
    evaluator_id: str
    profile_name: str
    status: str
    error_message: str | None
    evaluation_result: EvaluationResult


class EvaluateResponse(pydantic.BaseModel):
    results: list[EvaluateResult]


class ExportEvaluationResult(pydantic.BaseModel):
    app: str
    evaluator_id: str
    profile_name: str | None = None
    evaluated_model_system_prompt: str | None = None
    evaluated_model_retrieved_context: list[str] | None = None
    evaluated_model_input: str | None = None
    evaluated_model_output: str | None = None
    evaluated_model_gold_answer: str | None = None
    pass_: bool | None = pydantic.Field(alias="pass_", serialization_alias="pass")
    score_raw: float | None
    evaluation_duration: datetime.timedelta | None = None
    evaluated_model_name: str | None = None
    evaluated_model_provider: str | None = None
    evaluated_model_params: dict[str, str | int | float] | None = None
    evaluated_model_selected_model: str | None = None
    tags: dict[str, str] | None = None


class ExportEvaluationRequest(pydantic.BaseModel):
    evaluation_results: list[ExportEvaluationResult]


class ExportEvaluationResultPartial(pydantic.BaseModel):
    id: str
    app: str
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


class API(BaseAPIClient):
    async def evaluate(self, request: EvaluateRequest) -> EvaluateResponse:
        resp = await self.call("POST", "/v1/evaluate", body=request, response_cls=EvaluateResponse)
        # TODO error handling
        resp.response.raise_for_status()
        return resp.data

    async def export_evaluations(self, request: ExportEvaluationRequest) -> ExportEvaluationResponse:
        resp = await self.call(
            "POST",
            "/v1/evaluation-results/batch",
            body=request,
            response_cls=ExportEvaluationResponse,
        )
        # TODO error handling
        resp.response.raise_for_status()
        return resp.data

    async def get_profile(self, evaluator_family: str, name: str) -> EvaluatorProfile:
        profiles = await self.list_profiles(
            ListProfilesRequest(
                evaluator_family=evaluator_family,
                name=name,
                get_last_revision=True,
            )
        )
        # TODO error handling
        assert (
            len(profiles.evaluator_profiles) == 1
        ), f"get_profile didn't return 1 profile. It returned {len(profiles.evaluator_profiles)!r} instead"
        return profiles.evaluator_profiles[0]

    async def list_profiles(self, request: ListProfilesRequest) -> ListProfilesResponse:
        params = request.model_dump(exclude_none=True)
        resp = await self.call(
            "GET",
            "/v1/evaluator-profiles",
            params=params,
            response_cls=ListProfilesResponse,
        )
        # TODO error handling
        resp.response.raise_for_status()
        return resp.data

    async def list_dataset_data(self, dataset_id: str):
        resp = await self.call("GET", f"/v1/datasets/{dataset_id}/data", response_cls=ListDatasetData)
        # TODO error handling
        resp.response.raise_for_status()
        return resp.data
