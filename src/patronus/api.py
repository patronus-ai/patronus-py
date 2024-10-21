import datetime
import logging
from typing import Optional

from .api_base import BaseAPIClient, UnrecoverableAPIError, APIError, RPMLimitError
from . import api_types

log = logging.getLogger(__name__)


class API(BaseAPIClient):
    async def whoami(self) -> api_types.WhoAmIResponse:
        resp = await self.call("GET", "/v1/whoami", response_cls=api_types.WhoAmIResponse)
        resp.raise_for_status()
        return resp.data

    async def create_project(self, request: api_types.CreateProjectRequest) -> api_types.Project:
        resp = await self.call("POST", "/v1/projects", body=request, response_cls=api_types.Project)
        resp.raise_for_status()
        return resp.data

    async def get_project(self, project_id: str) -> api_types.Project:
        resp = await self.call("GET", f"/v1/projects/{project_id}", response_cls=api_types.GetProjectResponse)
        resp.raise_for_status()
        return resp.data.project

    async def create_experiment(self, request: api_types.CreateExperimentRequest) -> api_types.Experiment:
        resp = await self.call("POST", "/v1/experiments", body=request, response_cls=api_types.CreateExperimentResponse)
        resp.raise_for_status()
        return resp.data.experiment

    async def get_experiment(self, experiment_id: str) -> Optional[api_types.Experiment]:
        resp = await self.call(
            "GET",
            f"/v1/experiments/{experiment_id}",
            response_cls=api_types.GetExperimentResponse,
        )
        if resp.response.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.data.experiment

    async def evaluate(self, request: api_types.EvaluateRequest) -> api_types.EvaluateResponse:
        resp = await self.call("POST", "/v1/evaluate", body=request, response_cls=api_types.EvaluateResponse)

        # We set defaults in case ratelimits headers were not returned. It may happen in case of an error response,
        # or in rare cases like proxy stripping response headers.
        # The defaults are selected to proceed and fallback to standard retry mechanism.
        rpm_limit = try_int(resp.response.headers.get("x-ratelimit-rpm-limit-requests"), -1)
        rpm_remaining = try_int(resp.response.headers.get("x-ratelimit-rpm-remaining-requests"), 1)
        monthly_limit = try_int(resp.response.headers.get("x-ratelimit-monthly-limit-requests"), -1)
        monthly_remaining = try_int(resp.response.headers.get("x-ratelimit-monthly-remaining-requests"), 1)

        if resp.response.is_error:
            if resp.response.status_code == 429 and monthly_remaining <= 0:
                raise UnrecoverableAPIError(
                    f"Monthly evaluation {monthly_limit!r} limit hit",
                    response=resp.response,
                )
            if resp.response.status_code == 429 and rpm_remaining <= 0:
                wait_for_s = None
                try:
                    val: str = resp.response.headers.get("date")
                    response_date = datetime.datetime.strptime(val, "%a, %d %b %Y %H:%M:%S %Z")
                    wait_for_s = 60 - response_date.second
                except Exception as err:  # noqa
                    log.debug(
                        "Failed to extract RPM period from the response; "
                        f"'date' header value {resp.response.headers.get('date')!r}: "
                        f"{err}"
                    )
                    pass
                raise RPMLimitError(
                    limit=rpm_limit,
                    wait_for_s=wait_for_s,
                    response=resp.response,
                )
            # Generally, we assume that any 4xx error (excluding 429) is a user error
            # And repeated calls won't be successful.
            # 429 is an exception, but it should be handled above,
            # and if it's not then it should be handled as recoverable error.
            # It may not be handled above in rare cases - e.g. header is stripped by a proxy.
            if resp.response.status_code != 429 and resp.response.status_code < 500:
                raise UnrecoverableAPIError(
                    f"Response with unexpected status code: {resp.response.status_code}",
                    response=resp.response,
                )
            raise APIError(
                f"Response with unexpected status code: {resp.response.status_code}",
                response=resp.response,
            )

        for res in resp.data.results:
            if res.status == "validation_error":
                raise UnrecoverableAPIError("", response=resp.response)
            if res.status != "success":
                raise APIError(f"evaluation failed with status {res.status!r} and message {res.error_message!r}'")

        return resp.data

    async def export_evaluations(
        self, request: api_types.ExportEvaluationRequest
    ) -> api_types.ExportEvaluationResponse:
        resp = await self.call(
            "POST",
            "/v1/evaluation-results/batch",
            body=request,
            response_cls=api_types.ExportEvaluationResponse,
        )
        resp.raise_for_status()
        return resp.data

    async def list_evaluators(self) -> list[api_types.Evaluator]:
        resp = await self.call("GET", "/v1/evaluators", response_cls=api_types.ListEvaluatorsResponse)
        resp.raise_for_status()
        return resp.data.evaluators

    async def create_profile(self, request: api_types.CreateProfileRequest) -> api_types.CreateProfileResponse:
        resp = await self.call(
            "POST",
            "/v1/evaluator-profiles",
            body=request,
            response_cls=api_types.CreateProfileResponse,
        )
        resp.raise_for_status()
        return resp.data

    async def add_evaluator_profile_revision(
        self, evaluator_profile_id, request: api_types.AddEvaluatorProfileRevisionRequest
    ) -> api_types.AddEvaluatorProfileRevisionResponse:
        resp = await self.call(
            "POST",
            f"/v1/evaluator-profiles/{evaluator_profile_id}/revision",
            body=request,
            response_cls=api_types.AddEvaluatorProfileRevisionResponse,
        )
        resp.raise_for_status()
        return resp.data

    async def list_profiles(self, request: api_types.ListProfilesRequest) -> api_types.ListProfilesResponse:
        params = request.model_dump(exclude_none=True)
        resp = await self.call(
            "GET",
            "/v1/evaluator-profiles",
            params=params,
            response_cls=api_types.ListProfilesResponse,
        )
        resp.raise_for_status()
        return resp.data

    async def list_dataset_data(self, dataset_id: str) -> api_types.ListDatasetData:
        resp = await self.call("GET", f"/v1/datasets/{dataset_id}/data", response_cls=api_types.ListDatasetData)
        resp.raise_for_status()
        return resp.data


def try_int(v, default: int) -> int:
    if not v:
        return default
    try:
        return int(v)
    except ValueError:
        return default
