import json
import datetime
import logging
import typing
from typing import Optional, Union

from . import api_types
from .api_client_base import APIError, BaseAPIClient, CallResponse, RPMLimitError, UnrecoverableAPIError

log = logging.getLogger("patronus.core")


class PatronusAPIClient(BaseAPIClient):
    async def whoami(self) -> api_types.WhoAmIResponse:
        """Fetches information about the authenticated user."""
        resp = await self.call("GET", "/v1/whoami", response_cls=api_types.WhoAmIResponse)
        resp.raise_for_status()
        return resp.data

    def whoami_sync(self) -> api_types.WhoAmIResponse:
        """Fetches information about the authenticated user."""
        resp = self.call_sync("GET", "/v1/whoami", response_cls=api_types.WhoAmIResponse)
        resp.raise_for_status()
        return resp.data

    async def create_project(self, request: api_types.CreateProjectRequest) -> api_types.Project:
        """Creates a new project based on the given request."""
        resp = await self.call("POST", "/v1/projects", body=request, response_cls=api_types.Project)
        resp.raise_for_status()
        return resp.data

    def create_project_sync(self, request: api_types.CreateProjectRequest) -> api_types.Project:
        """Creates a new project based on the given request."""
        resp = self.call_sync("POST", "/v1/projects", body=request, response_cls=api_types.Project)
        resp.raise_for_status()
        return resp.data

    async def get_project(self, project_id: str) -> api_types.Project:
        """Fetches a project by its ID."""
        resp = await self.call(
            "GET",
            f"/v1/projects/{project_id}",
            response_cls=api_types.GetProjectResponse,
        )
        resp.raise_for_status()
        return resp.data.project

    def get_project_sync(self, project_id: str) -> api_types.Project:
        """Fetches a project by its ID."""
        resp = self.call_sync(
            "GET",
            f"/v1/projects/{project_id}",
            response_cls=api_types.GetProjectResponse,
        )
        resp.raise_for_status()
        return resp.data.project

    async def create_experiment(self, request: api_types.CreateExperimentRequest) -> api_types.Experiment:
        """Creates a new experiment based on the given request."""
        resp = await self.call(
            "POST",
            "/v1/experiments",
            body=request,
            response_cls=api_types.CreateExperimentResponse,
        )
        resp.raise_for_status()
        return resp.data.experiment

    def create_experiment_sync(self, request: api_types.CreateExperimentRequest) -> api_types.Experiment:
        """Creates a new experiment based on the given request."""
        resp = self.call_sync(
            "POST",
            "/v1/experiments",
            body=request,
            response_cls=api_types.CreateExperimentResponse,
        )
        resp.raise_for_status()
        return resp.data.experiment

    async def update_experiment(
        self, experiment_id: str, request: api_types.UpdateExperimentRequest
    ) -> api_types.Experiment:
        """Updates an existing experiment based on the given request."""
        resp = await self.call(
            "POST",
            f"/v1/experiments/{experiment_id}",
            body=request,
            response_cls=api_types.UpdateExperimentResponse,
        )
        resp.raise_for_status()
        return resp.data.experiment

    def update_experiment_sync(
        self, experiment_id: str, request: api_types.UpdateExperimentRequest
    ) -> api_types.Experiment:
        """Updates an existing experiment based on the given request."""
        resp = self.call_sync(
            "POST",
            f"/v1/experiments{experiment_id}",
            body=request,
            response_cls=api_types.UpdateExperimentResponse,
        )
        resp.raise_for_status()
        return resp.data.experiment

    async def get_experiment(self, experiment_id: str) -> Optional[api_types.Experiment]:
        """Fetches an experiment by its ID or returns None if not found."""
        resp = await self.call(
            "GET",
            f"/v1/experiments/{experiment_id}",
            response_cls=api_types.GetExperimentResponse,
        )
        if resp.response.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.data.experiment

    def get_experiment_sync(self, experiment_id: str) -> Optional[api_types.Experiment]:
        """Fetches an experiment by its ID or returns None if not found."""
        resp = self.call_sync(
            "GET",
            f"/v1/experiments/{experiment_id}",
            response_cls=api_types.GetExperimentResponse,
        )
        if resp.response.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.data.experiment

    async def evaluate(self, request: api_types.EvaluateRequest) -> api_types.EvaluateResponse:
        """Evaluates content using the specified evaluators."""
        resp = await self.call(
            "POST",
            "/v1/evaluate",
            body=request,
            response_cls=api_types.EvaluateResponse,
        )
        resp.raise_for_status()
        return resp.data

    def evaluate_sync(self, request: api_types.EvaluateRequest) -> api_types.EvaluateResponse:
        """Evaluates content using the specified evaluators."""
        resp = self.call_sync(
            "POST",
            "/v1/evaluate",
            body=request,
            response_cls=api_types.EvaluateResponse,
        )
        resp.raise_for_status()
        return resp.data

    async def evaluate_one(self, request: api_types.EvaluateRequest) -> api_types.EvaluationResult:
        """Evaluates content using a single evaluator."""
        if len(request.evaluators) > 1:
            raise ValueError("'evaluate_one()' cannot accept more than one evaluator in the request body")
        resp = await self.call(
            "POST",
            "/v1/evaluate",
            body=request,
            response_cls=api_types.EvaluateResponse,
        )
        return self._evaluate_one_process_resp(resp)

    def evaluate_one_sync(self, request: api_types.EvaluateRequest) -> api_types.EvaluationResult:
        """Evaluates content using a single evaluator."""
        if len(request.evaluators) > 1:
            raise ValueError("'evaluate_one_sync()' cannot accept more than one evaluator in the request body")
        resp = self.call_sync(
            "POST",
            "/v1/evaluate",
            body=request,
            response_cls=api_types.EvaluateResponse,
        )
        return self._evaluate_one_process_resp(resp)

    @staticmethod
    def _evaluate_one_process_resp(
        resp: CallResponse[api_types.EvaluateResponse],
    ) -> api_types.EvaluationResult:
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

        return resp.data.results[0].evaluation_result

    async def export_evaluations(
        self, request: api_types.ExportEvaluationRequest
    ) -> api_types.ExportEvaluationResponse:
        """Exports evaluations based on the given request."""
        resp = await self.call(
            "POST",
            "/v1/evaluation-results/batch",
            body=request,
            response_cls=api_types.ExportEvaluationResponse,
        )
        resp.raise_for_status()
        return resp.data

    def export_evaluations_sync(self, request: api_types.ExportEvaluationRequest) -> api_types.ExportEvaluationResponse:
        """Exports evaluations based on the given request."""
        resp = self.call_sync(
            "POST",
            "/v1/evaluation-results/batch",
            body=request,
            response_cls=api_types.ExportEvaluationResponse,
        )
        resp.raise_for_status()
        return resp.data

    async def list_evaluators(self, by_alias_or_id: Optional[str] = None) -> list[api_types.Evaluator]:
        """Retrieves a list of available evaluators."""
        params = {}
        if by_alias_or_id:
            params["by_alias_or_id"] = by_alias_or_id

        resp = await self.call("GET", "/v1/evaluators", params=params, response_cls=api_types.ListEvaluatorsResponse)
        resp.raise_for_status()
        return resp.data.evaluators

    def list_evaluators_sync(self, by_alias_or_id: Optional[str] = None) -> list[api_types.Evaluator]:
        """Retrieves a list of available evaluators."""
        params = {}
        if by_alias_or_id:
            params["by_alias_or_id"] = by_alias_or_id

        resp = self.call_sync("GET", "/v1/evaluators", params=params, response_cls=api_types.ListEvaluatorsResponse)
        resp.raise_for_status()
        return resp.data.evaluators

    async def create_criteria(self, request: api_types.CreateCriteriaRequest) -> api_types.CreateCriteriaResponse:
        """Creates evaluation criteria based on the given request."""
        resp = await self.call(
            "POST",
            "/v1/evaluator-criteria",
            body=request,
            response_cls=api_types.CreateCriteriaResponse,
        )
        resp.raise_for_status()
        return resp.data

    def create_criteria_sync(self, request: api_types.CreateCriteriaRequest) -> api_types.CreateCriteriaResponse:
        """Creates evaluation criteria based on the given request."""
        resp = self.call_sync(
            "POST",
            "/v1/evaluator-criteria",
            body=request,
            response_cls=api_types.CreateCriteriaResponse,
        )
        resp.raise_for_status()
        return resp.data

    async def add_evaluator_criteria_revision(
        self,
        evaluator_criteria_id,
        request: api_types.AddEvaluatorCriteriaRevisionRequest,
    ) -> api_types.AddEvaluatorCriteriaRevisionResponse:
        """Adds a revision to existing evaluator criteria."""
        resp = await self.call(
            "POST",
            f"/v1/evaluator-criteria/{evaluator_criteria_id}/revision",
            body=request,
            response_cls=api_types.AddEvaluatorCriteriaRevisionResponse,
        )
        resp.raise_for_status()
        return resp.data

    def add_evaluator_criteria_revision_sync(
        self,
        evaluator_criteria_id,
        request: api_types.AddEvaluatorCriteriaRevisionRequest,
    ) -> api_types.AddEvaluatorCriteriaRevisionResponse:
        """Adds a revision to existing evaluator criteria."""
        resp = self.call_sync(
            "POST",
            f"/v1/evaluator-criteria/{evaluator_criteria_id}/revision",
            body=request,
            response_cls=api_types.AddEvaluatorCriteriaRevisionResponse,
        )
        resp.raise_for_status()
        return resp.data

    async def list_criteria(self, request: api_types.ListCriteriaRequest) -> api_types.ListCriteriaResponse:
        """Retrieves a list of evaluation criteria based on the given request."""
        params = request.model_dump(exclude_none=True)
        resp = await self.call(
            "GET",
            "/v1/evaluator-criteria",
            params=params,
            response_cls=api_types.ListCriteriaResponse,
        )
        resp.raise_for_status()
        return resp.data

    def list_criteria_sync(self, request: api_types.ListCriteriaRequest) -> api_types.ListCriteriaResponse:
        """Retrieves a list of evaluation criteria based on the given request."""
        params = request.model_dump(exclude_none=True)
        resp = self.call_sync(
            "GET",
            "/v1/evaluator-criteria",
            params=params,
            response_cls=api_types.ListCriteriaResponse,
        )
        resp.raise_for_status()
        return resp.data

    async def list_datasets(self, dataset_type: Optional[str] = None) -> list[api_types.Dataset]:
        """
        Retrieves a list of datasets, optionally filtered by type.
        """
        params = {}
        if dataset_type is not None:
            params["type"] = dataset_type

        resp = await self.call(
            "GET",
            "/v1/datasets",
            params=params,
            response_cls=api_types.ListDatasetsResponse,
        )
        resp.raise_for_status()
        return resp.data.datasets

    def list_datasets_sync(self, dataset_type: Optional[str] = None) -> list[api_types.Dataset]:
        """
        Retrieves a list of datasets, optionally filtered by type.
        """
        params = {}
        if dataset_type is not None:
            params["type"] = dataset_type

        resp = self.call_sync(
            "GET",
            "/v1/datasets",
            params=params,
            response_cls=api_types.ListDatasetsResponse,
        )
        resp.raise_for_status()
        return resp.data.datasets

    async def list_dataset_data(self, dataset_id: str) -> api_types.ListDatasetData:
        """Retrieves data from a dataset by its ID."""
        resp = await self.call(
            "GET",
            f"/v1/datasets/{dataset_id}/data",
            response_cls=api_types.ListDatasetData,
        )
        resp.raise_for_status()
        return resp.data

    def list_dataset_data_sync(self, dataset_id: str) -> api_types.ListDatasetData:
        """Retrieves data from a dataset by its ID."""
        resp = self.call_sync(
            "GET",
            f"/v1/datasets/{dataset_id}/data",
            response_cls=api_types.ListDatasetData,
        )
        resp.raise_for_status()
        return resp.data

    async def upload_dataset(
        self,
        file_path: str,
        dataset_name: str,
        dataset_description: Optional[str] = None,
        custom_field_mapping: Optional[dict[str, Union[str, list[str]]]] = None,
    ) -> api_types.Dataset:
        """
        Upload a dataset file to create a new dataset in Patronus.

        Args:
            file_path: Path to the dataset file (CSV or JSONL format)
            dataset_name: Name for the created dataset
            dataset_description: Optional description for the dataset
            custom_field_mapping: Optional mapping of standard field names to custom field names in the dataset

        Returns:
            Dataset object representing the created dataset
        """
        with open(file_path, "rb") as f:
            return await self.upload_dataset_from_buffer(f, dataset_name, dataset_description, custom_field_mapping)

    async def upload_dataset_from_buffer(
        self,
        file_obj: typing.BinaryIO,
        dataset_name: str,
        dataset_description: Optional[str] = None,
        custom_field_mapping: Optional[dict[str, Union[str, list[str]]]] = None,
    ) -> api_types.Dataset:
        """
        Upload a dataset file to create a new dataset in Patronus AI Platform.

        Args:
            file_obj: File-like object containing dataset content (CSV or JSONL format)
            dataset_name: Name for the created dataset
            dataset_description: Optional description for the dataset
            custom_field_mapping: Optional mapping of standard field names to custom field names in the dataset

        Returns:
            Dataset object representing the created dataset
        """
        data = {
            "dataset_name": dataset_name,
        }

        if dataset_description is not None:
            data["dataset_description"] = dataset_description

        if custom_field_mapping is not None:
            data["custom_field_mapping"] = json.dumps(custom_field_mapping)

        files = {"file": (dataset_name, file_obj)}

        resp = await self.call_multipart(
            "POST",
            "/v1/datasets",
            files=files,
            data=data,
            response_cls=api_types.CreateDatasetResponse,
        )

        resp.raise_for_status()
        return resp.data.dataset

    def upload_dataset_sync(
        self,
        file_path: str,
        dataset_name: str,
        dataset_description: Optional[str] = None,
        custom_field_mapping: Optional[dict[str, Union[str, list[str]]]] = None,
    ) -> api_types.Dataset:
        """
        Upload a dataset file to create a new dataset in Patronus AI Platform.

        Args:
            file_path: Path to the dataset file (CSV or JSONL format)
            dataset_name: Name for the created dataset
            dataset_description: Optional description for the dataset
            custom_field_mapping: Optional mapping of standard field names to custom field names in the dataset

        Returns:
            Dataset object representing the created dataset
        """
        with open(file_path, "rb") as f:
            return self.upload_dataset_from_buffer_sync(f, dataset_name, dataset_description, custom_field_mapping)

    def upload_dataset_from_buffer_sync(
        self,
        file_obj: typing.BinaryIO,
        dataset_name: str,
        dataset_description: Optional[str] = None,
        custom_field_mapping: Optional[dict[str, Union[str, list[str]]]] = None,
    ) -> api_types.Dataset:
        """
        Upload a dataset file to create a new dataset in Patronus AI Platform.

        Args:
            file_obj: File-like object containing dataset content (CSV or JSONL format)
            dataset_name: Name for the created dataset
            dataset_description: Optional description for the dataset
            custom_field_mapping: Optional mapping of standard field names to custom field names in the dataset

        Returns:
            Dataset object representing the created dataset
        """
        data = {
            "dataset_name": dataset_name,
        }

        if dataset_description is not None:
            data["dataset_description"] = dataset_description

        if custom_field_mapping is not None:
            data["custom_field_mapping"] = json.dumps(custom_field_mapping)

        files = {"file": (dataset_name, file_obj)}

        resp = self.call_multipart_sync(
            "POST",
            "/v1/datasets",
            files=files,
            data=data,
            response_cls=api_types.CreateDatasetResponse,
        )

        resp.raise_for_status()
        return resp.data.dataset

    async def batch_create_evaluations(
        self, request: api_types.BatchCreateEvaluationsRequest
    ) -> api_types.BatchCreateEvaluationsResponse:
        """Creates multiple evaluations in a single request."""
        resp = await self.call(
            "POST",
            "/v1/evaluations/batch",
            body=request,
            response_cls=api_types.BatchCreateEvaluationsResponse,
        )
        resp.raise_for_status()
        return resp.data

    def batch_create_evaluations_sync(
        self, request: api_types.BatchCreateEvaluationsRequest
    ) -> api_types.BatchCreateEvaluationsResponse:
        """Creates multiple evaluations in a single request."""
        resp = self.call_sync(
            "POST",
            "/v1/evaluations/batch",
            body=request,
            response_cls=api_types.BatchCreateEvaluationsResponse,
        )
        resp.raise_for_status()
        return resp.data

    def search_evaluations_sync(
        self, request: api_types.SearchEvaluationsRequest
    ) -> api_types.SearchEvaluationsResponse:
        """Searches for evaluations based on the given criteria."""
        resp = self.call_sync(
            "POST",
            "/v1/evaluations/search",
            body=request,
            response_cls=api_types.SearchEvaluationsResponse,
        )
        resp.raise_for_status()
        return resp.data

    async def search_evaluations(
        self, request: api_types.SearchEvaluationsRequest
    ) -> api_types.SearchEvaluationsResponse:
        """Searches for evaluations based on the given criteria."""
        resp = await self.call(
            "POST",
            "/v1/evaluations/search",
            body=request,
            response_cls=api_types.SearchEvaluationsResponse,
        )
        resp.raise_for_status()
        return resp.data

    async def annotate(self, request: api_types.AnnotateRequest) -> api_types.AnnotateResponse:
        """Annotates log based on the given request."""
        resp = await self.call(
            "POST",
            "/v1/annotate",
            body=request,
            response_cls=api_types.AnnotateResponse,
        )
        resp.raise_for_status()
        return resp.data

    def annotate_sync(self, request: api_types.AnnotateRequest) -> api_types.AnnotateResponse:
        """Annotates log based on the given request."""
        resp = self.call_sync(
            "POST",
            "/v1/annotate",
            body=request,
            response_cls=api_types.AnnotateResponse,
        )
        resp.raise_for_status()
        return resp.data

    async def list_annotation_criteria(
        self, *, project_id: Optional[str] = None, limit: Optional[int] = None, offset: Optional[int] = None
    ) -> api_types.ListAnnotationCriteriaResponse:
        """Retrieves a list of annotation criteria with optional filtering."""
        params = {}
        if project_id is not None:
            params["project_id"] = project_id
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset
        resp = await self.call(
            "GET",
            "/v1/annotation-criteria",
            params=params,
            response_cls=api_types.ListAnnotationCriteriaResponse,
        )
        resp.raise_for_status()
        return resp.data

    def list_annotation_criteria_sync(
        self, *, project_id: Optional[str] = None, limit: Optional[int] = None, offset: Optional[int] = None
    ) -> api_types.ListAnnotationCriteriaResponse:
        """Retrieves a list of annotation criteria with optional filtering."""
        params = {}
        if project_id is not None:
            params["project_id"] = project_id
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset
        resp = self.call_sync(
            "GET",
            "/v1/annotation-criteria",
            params=params,
            response_cls=api_types.ListAnnotationCriteriaResponse,
        )
        resp.raise_for_status()
        return resp.data

    async def create_annotation_criteria(
        self, request: api_types.CreateAnnotationCriteriaRequest
    ) -> api_types.CreateAnnotationCriteriaResponse:
        """Creates annotation criteria based on the given request."""
        resp = await self.call(
            "POST",
            "/v1/annotation-criteria",
            body=request,
            response_cls=api_types.CreateAnnotationCriteriaResponse,
        )
        resp.raise_for_status()
        return resp.data

    def create_annotation_criteria_sync(
        self, request: api_types.CreateAnnotationCriteriaRequest
    ) -> api_types.CreateAnnotationCriteriaResponse:
        """Creates annotation criteria based on the given request."""
        resp = self.call_sync(
            "POST",
            "/v1/annotation-criteria",
            body=request,
            response_cls=api_types.CreateAnnotationCriteriaResponse,
        )
        resp.raise_for_status()
        return resp.data

    async def update_annotation_criteria(
        self, criteria_id: str, request: api_types.UpdateAnnotationCriteriaRequest
    ) -> api_types.UpdateAnnotationCriteriaResponse:
        """Creates annotation criteria based on the given request."""
        resp = await self.call(
            "PUT",
            f"/v1/annotation-criteria/{criteria_id}",
            body=request,
            response_cls=api_types.UpdateAnnotationCriteriaResponse,
        )
        resp.raise_for_status()
        return resp.data

    def update_annotation_criteria_sync(
        self, criteria_id: str, request: api_types.UpdateAnnotationCriteriaRequest
    ) -> api_types.UpdateAnnotationCriteriaResponse:
        """Creates annotation criteria based on the given request."""
        resp = self.call_sync(
            "PUT",
            f"/v1/annotation-criteria/{criteria_id}",
            body=request,
            response_cls=api_types.UpdateAnnotationCriteriaResponse,
        )
        resp.raise_for_status()
        return resp.data

    async def delete_annotation_criteria(self, criteria_id: str) -> None:
        """Deletes annotation criteria by its ID."""
        resp = await self.call(
            "DELETE",
            f"/v1/annotation-criteria/{criteria_id}",
            response_cls=None,
        )
        resp.raise_for_status()

    def delete_annotation_criteria_sync(self, criteria_id: str) -> None:
        """Deletes annotation criteria by its ID."""
        resp = self.call_sync(
            "DELETE",
            f"/v1/annotation-criteria/{criteria_id}",
            response_cls=None,
        )
        resp.raise_for_status()

    async def search_logs(self, request: api_types.SearchLogsRequest) -> api_types.SearchLogsResponse:
        """Searches for logs based on the given request."""
        resp = await self.call(
            "POST",
            "/v1/otel/logs/search",
            body=request,
            response_cls=api_types.SearchLogsResponse,
        )
        resp.raise_for_status()
        return resp.data

    def search_logs_sync(self, request: api_types.SearchLogsRequest) -> api_types.SearchLogsResponse:
        """Searches for logs based on the given request."""
        resp = self.call_sync(
            "POST",
            "/v1/otel/logs/search",
            body=request,
            response_cls=api_types.SearchLogsResponse,
        )
        resp.raise_for_status()
        return resp.data


def try_int(v, default: int) -> int:
    if not v:
        return default
    try:
        return int(v)
    except ValueError:
        return default
