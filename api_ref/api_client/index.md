# API

## patronus.api.api_client.PatronusAPIClient

```python
PatronusAPIClient(
    *,
    client_http_async: AsyncClient,
    client_http: Client,
    base_url: str,
    api_key: str,
)

```

Bases: `BaseAPIClient`

Source code in `src/patronus/api/api_client_base.py`

```python
def __init__(
    self,
    *,
    client_http_async: httpx.AsyncClient,
    client_http: httpx.Client,
    base_url: str,
    api_key: str,
):
    self.version = importlib.metadata.version("patronus")
    self.http = client_http_async
    self.http_sync = client_http
    self.base_url = base_url.rstrip("/")
    self.api_key = api_key

```

### add_evaluator_criteria_revision

```python
add_evaluator_criteria_revision(
    evaluator_criteria_id,
    request: AddEvaluatorCriteriaRevisionRequest,
) -> api_types.AddEvaluatorCriteriaRevisionResponse

```

Adds a revision to existing evaluator criteria.

Source code in `src/patronus/api/api_client.py`

```python
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

```

### add_evaluator_criteria_revision_sync

```python
add_evaluator_criteria_revision_sync(
    evaluator_criteria_id,
    request: AddEvaluatorCriteriaRevisionRequest,
) -> api_types.AddEvaluatorCriteriaRevisionResponse

```

Adds a revision to existing evaluator criteria.

Source code in `src/patronus/api/api_client.py`

```python
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

```

### annotate

```python
annotate(
    request: AnnotateRequest,
) -> api_types.AnnotateResponse

```

Annotates log based on the given request.

Source code in `src/patronus/api/api_client.py`

```python
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

```

### annotate_sync

```python
annotate_sync(
    request: AnnotateRequest,
) -> api_types.AnnotateResponse

```

Annotates log based on the given request.

Source code in `src/patronus/api/api_client.py`

```python
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

```

### batch_create_evaluations

```python
batch_create_evaluations(
    request: BatchCreateEvaluationsRequest,
) -> api_types.BatchCreateEvaluationsResponse

```

Creates multiple evaluations in a single request.

Source code in `src/patronus/api/api_client.py`

```python
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

```

### batch_create_evaluations_sync

```python
batch_create_evaluations_sync(
    request: BatchCreateEvaluationsRequest,
) -> api_types.BatchCreateEvaluationsResponse

```

Creates multiple evaluations in a single request.

Source code in `src/patronus/api/api_client.py`

```python
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

```

### create_annotation_criteria

```python
create_annotation_criteria(
    request: CreateAnnotationCriteriaRequest,
) -> api_types.CreateAnnotationCriteriaResponse

```

Creates annotation criteria based on the given request.

Source code in `src/patronus/api/api_client.py`

```python
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

```

### create_annotation_criteria_sync

```python
create_annotation_criteria_sync(
    request: CreateAnnotationCriteriaRequest,
) -> api_types.CreateAnnotationCriteriaResponse

```

Creates annotation criteria based on the given request.

Source code in `src/patronus/api/api_client.py`

```python
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

```

### create_criteria

```python
create_criteria(
    request: CreateCriteriaRequest,
) -> api_types.CreateCriteriaResponse

```

Creates evaluation criteria based on the given request.

Source code in `src/patronus/api/api_client.py`

```python
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

```

### create_criteria_sync

```python
create_criteria_sync(
    request: CreateCriteriaRequest,
) -> api_types.CreateCriteriaResponse

```

Creates evaluation criteria based on the given request.

Source code in `src/patronus/api/api_client.py`

```python
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

```

### create_experiment

```python
create_experiment(
    request: CreateExperimentRequest,
) -> api_types.Experiment

```

Creates a new experiment based on the given request.

Source code in `src/patronus/api/api_client.py`

```python
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

```

### create_experiment_sync

```python
create_experiment_sync(
    request: CreateExperimentRequest,
) -> api_types.Experiment

```

Creates a new experiment based on the given request.

Source code in `src/patronus/api/api_client.py`

```python
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

```

### create_project

```python
create_project(
    request: CreateProjectRequest,
) -> api_types.Project

```

Creates a new project based on the given request.

Source code in `src/patronus/api/api_client.py`

```python
async def create_project(self, request: api_types.CreateProjectRequest) -> api_types.Project:
    """Creates a new project based on the given request."""
    resp = await self.call("POST", "/v1/projects", body=request, response_cls=api_types.Project)
    resp.raise_for_status()
    return resp.data

```

### create_project_sync

```python
create_project_sync(
    request: CreateProjectRequest,
) -> api_types.Project

```

Creates a new project based on the given request.

Source code in `src/patronus/api/api_client.py`

```python
def create_project_sync(self, request: api_types.CreateProjectRequest) -> api_types.Project:
    """Creates a new project based on the given request."""
    resp = self.call_sync("POST", "/v1/projects", body=request, response_cls=api_types.Project)
    resp.raise_for_status()
    return resp.data

```

### delete_annotation_criteria

```python
delete_annotation_criteria(criteria_id: str) -> None

```

Deletes annotation criteria by its ID.

Source code in `src/patronus/api/api_client.py`

```python
async def delete_annotation_criteria(self, criteria_id: str) -> None:
    """Deletes annotation criteria by its ID."""
    resp = await self.call(
        "DELETE",
        f"/v1/annotation-criteria/{criteria_id}",
        response_cls=None,
    )
    resp.raise_for_status()

```

### delete_annotation_criteria_sync

```python
delete_annotation_criteria_sync(criteria_id: str) -> None

```

Deletes annotation criteria by its ID.

Source code in `src/patronus/api/api_client.py`

```python
def delete_annotation_criteria_sync(self, criteria_id: str) -> None:
    """Deletes annotation criteria by its ID."""
    resp = self.call_sync(
        "DELETE",
        f"/v1/annotation-criteria/{criteria_id}",
        response_cls=None,
    )
    resp.raise_for_status()

```

### evaluate

```python
evaluate(
    request: EvaluateRequest,
) -> api_types.EvaluateResponse

```

Evaluates content using the specified evaluators.

Source code in `src/patronus/api/api_client.py`

```python
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

```

### evaluate_one

```python
evaluate_one(
    request: EvaluateRequest,
) -> api_types.EvaluationResult

```

Evaluates content using a single evaluator.

Source code in `src/patronus/api/api_client.py`

```python
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

```

### evaluate_one_sync

```python
evaluate_one_sync(
    request: EvaluateRequest,
) -> api_types.EvaluationResult

```

Evaluates content using a single evaluator.

Source code in `src/patronus/api/api_client.py`

```python
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

```

### evaluate_sync

```python
evaluate_sync(
    request: EvaluateRequest,
) -> api_types.EvaluateResponse

```

Evaluates content using the specified evaluators.

Source code in `src/patronus/api/api_client.py`

```python
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

```

### export_evaluations

```python
export_evaluations(
    request: ExportEvaluationRequest,
) -> api_types.ExportEvaluationResponse

```

Exports evaluations based on the given request.

Source code in `src/patronus/api/api_client.py`

```python
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

```

### export_evaluations_sync

```python
export_evaluations_sync(
    request: ExportEvaluationRequest,
) -> api_types.ExportEvaluationResponse

```

Exports evaluations based on the given request.

Source code in `src/patronus/api/api_client.py`

```python
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

```

### get_experiment

```python
get_experiment(
    experiment_id: str,
) -> Optional[api_types.Experiment]

```

Fetches an experiment by its ID or returns None if not found.

Source code in `src/patronus/api/api_client.py`

```python
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

```

### get_experiment_sync

```python
get_experiment_sync(
    experiment_id: str,
) -> Optional[api_types.Experiment]

```

Fetches an experiment by its ID or returns None if not found.

Source code in `src/patronus/api/api_client.py`

```python
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

```

### get_project

```python
get_project(project_id: str) -> api_types.Project

```

Fetches a project by its ID.

Source code in `src/patronus/api/api_client.py`

```python
async def get_project(self, project_id: str) -> api_types.Project:
    """Fetches a project by its ID."""
    resp = await self.call(
        "GET",
        f"/v1/projects/{project_id}",
        response_cls=api_types.GetProjectResponse,
    )
    resp.raise_for_status()
    return resp.data.project

```

### get_project_sync

```python
get_project_sync(project_id: str) -> api_types.Project

```

Fetches a project by its ID.

Source code in `src/patronus/api/api_client.py`

```python
def get_project_sync(self, project_id: str) -> api_types.Project:
    """Fetches a project by its ID."""
    resp = self.call_sync(
        "GET",
        f"/v1/projects/{project_id}",
        response_cls=api_types.GetProjectResponse,
    )
    resp.raise_for_status()
    return resp.data.project

```

### list_annotation_criteria

```python
list_annotation_criteria(
    *,
    project_id: Optional[str] = None,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
) -> api_types.ListAnnotationCriteriaResponse

```

Retrieves a list of annotation criteria with optional filtering.

Source code in `src/patronus/api/api_client.py`

```python
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

```

### list_annotation_criteria_sync

```python
list_annotation_criteria_sync(
    *,
    project_id: Optional[str] = None,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
) -> api_types.ListAnnotationCriteriaResponse

```

Retrieves a list of annotation criteria with optional filtering.

Source code in `src/patronus/api/api_client.py`

```python
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

```

### list_criteria

```python
list_criteria(
    request: ListCriteriaRequest,
) -> api_types.ListCriteriaResponse

```

Retrieves a list of evaluation criteria based on the given request.

Source code in `src/patronus/api/api_client.py`

```python
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

```

### list_criteria_sync

```python
list_criteria_sync(
    request: ListCriteriaRequest,
) -> api_types.ListCriteriaResponse

```

Retrieves a list of evaluation criteria based on the given request.

Source code in `src/patronus/api/api_client.py`

```python
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

```

### list_dataset_data

```python
list_dataset_data(
    dataset_id: str,
) -> api_types.ListDatasetData

```

Retrieves data from a dataset by its ID.

Source code in `src/patronus/api/api_client.py`

```python
async def list_dataset_data(self, dataset_id: str) -> api_types.ListDatasetData:
    """Retrieves data from a dataset by its ID."""
    resp = await self.call(
        "GET",
        f"/v1/datasets/{dataset_id}/data",
        response_cls=api_types.ListDatasetData,
    )
    resp.raise_for_status()
    return resp.data

```

### list_dataset_data_sync

```python
list_dataset_data_sync(
    dataset_id: str,
) -> api_types.ListDatasetData

```

Retrieves data from a dataset by its ID.

Source code in `src/patronus/api/api_client.py`

```python
def list_dataset_data_sync(self, dataset_id: str) -> api_types.ListDatasetData:
    """Retrieves data from a dataset by its ID."""
    resp = self.call_sync(
        "GET",
        f"/v1/datasets/{dataset_id}/data",
        response_cls=api_types.ListDatasetData,
    )
    resp.raise_for_status()
    return resp.data

```

### list_datasets

```python
list_datasets(
    dataset_type: Optional[str] = None,
) -> list[api_types.Dataset]

```

Retrieves a list of datasets, optionally filtered by type.

Source code in `src/patronus/api/api_client.py`

```python
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

```

### list_datasets_sync

```python
list_datasets_sync(
    dataset_type: Optional[str] = None,
) -> list[api_types.Dataset]

```

Retrieves a list of datasets, optionally filtered by type.

Source code in `src/patronus/api/api_client.py`

```python
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

```

### list_evaluators

```python
list_evaluators(
    by_alias_or_id: Optional[str] = None,
) -> list[api_types.Evaluator]

```

Retrieves a list of available evaluators.

Source code in `src/patronus/api/api_client.py`

```python
async def list_evaluators(self, by_alias_or_id: Optional[str] = None) -> list[api_types.Evaluator]:
    """Retrieves a list of available evaluators."""
    params = {}
    if by_alias_or_id:
        params["by_alias_or_id"] = by_alias_or_id

    resp = await self.call("GET", "/v1/evaluators", params=params, response_cls=api_types.ListEvaluatorsResponse)
    resp.raise_for_status()
    return resp.data.evaluators

```

### list_evaluators_sync

```python
list_evaluators_sync(
    by_alias_or_id: Optional[str] = None,
) -> list[api_types.Evaluator]

```

Retrieves a list of available evaluators.

Source code in `src/patronus/api/api_client.py`

```python
def list_evaluators_sync(self, by_alias_or_id: Optional[str] = None) -> list[api_types.Evaluator]:
    """Retrieves a list of available evaluators."""
    params = {}
    if by_alias_or_id:
        params["by_alias_or_id"] = by_alias_or_id

    resp = self.call_sync("GET", "/v1/evaluators", params=params, response_cls=api_types.ListEvaluatorsResponse)
    resp.raise_for_status()
    return resp.data.evaluators

```

### search_evaluations

```python
search_evaluations(
    request: SearchEvaluationsRequest,
) -> api_types.SearchEvaluationsResponse

```

Searches for evaluations based on the given criteria.

Source code in `src/patronus/api/api_client.py`

```python
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

```

### search_evaluations_sync

```python
search_evaluations_sync(
    request: SearchEvaluationsRequest,
) -> api_types.SearchEvaluationsResponse

```

Searches for evaluations based on the given criteria.

Source code in `src/patronus/api/api_client.py`

```python
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

```

### search_logs

```python
search_logs(
    request: SearchLogsRequest,
) -> api_types.SearchLogsResponse

```

Searches for logs based on the given request.

Source code in `src/patronus/api/api_client.py`

```python
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

```

### search_logs_sync

```python
search_logs_sync(
    request: SearchLogsRequest,
) -> api_types.SearchLogsResponse

```

Searches for logs based on the given request.

Source code in `src/patronus/api/api_client.py`

```python
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

```

### update_annotation_criteria

```python
update_annotation_criteria(
    criteria_id: str,
    request: UpdateAnnotationCriteriaRequest,
) -> api_types.UpdateAnnotationCriteriaResponse

```

Creates annotation criteria based on the given request.

Source code in `src/patronus/api/api_client.py`

```python
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

```

### update_annotation_criteria_sync

```python
update_annotation_criteria_sync(
    criteria_id: str,
    request: UpdateAnnotationCriteriaRequest,
) -> api_types.UpdateAnnotationCriteriaResponse

```

Creates annotation criteria based on the given request.

Source code in `src/patronus/api/api_client.py`

```python
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

```

### update_experiment

```python
update_experiment(
    experiment_id: str, request: UpdateExperimentRequest
) -> api_types.Experiment

```

Updates an existing experiment based on the given request.

Source code in `src/patronus/api/api_client.py`

```python
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

```

### update_experiment_sync

```python
update_experiment_sync(
    experiment_id: str, request: UpdateExperimentRequest
) -> api_types.Experiment

```

Updates an existing experiment based on the given request.

Source code in `src/patronus/api/api_client.py`

```python
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

```

### upload_dataset

```python
upload_dataset(
    file_path: str,
    dataset_name: str,
    dataset_description: Optional[str] = None,
    custom_field_mapping: Optional[
        dict[str, Union[str, list[str]]]
    ] = None,
) -> api_types.Dataset

```

Upload a dataset file to create a new dataset in Patronus.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `file_path` | `str` | Path to the dataset file (CSV or JSONL format) | *required* | | `dataset_name` | `str` | Name for the created dataset | *required* | | `dataset_description` | `Optional[str]` | Optional description for the dataset | `None` | | `custom_field_mapping` | `Optional[dict[str, Union[str, list[str]]]]` | Optional mapping of standard field names to custom field names in the dataset | `None` |

Returns:

| Type | Description | | --- | --- | | `Dataset` | Dataset object representing the created dataset |

Source code in `src/patronus/api/api_client.py`

```python
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

```

### upload_dataset_from_buffer

```python
upload_dataset_from_buffer(
    file_obj: BinaryIO,
    dataset_name: str,
    dataset_description: Optional[str] = None,
    custom_field_mapping: Optional[
        dict[str, Union[str, list[str]]]
    ] = None,
) -> api_types.Dataset

```

Upload a dataset file to create a new dataset in Patronus AI Platform.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `file_obj` | `BinaryIO` | File-like object containing dataset content (CSV or JSONL format) | *required* | | `dataset_name` | `str` | Name for the created dataset | *required* | | `dataset_description` | `Optional[str]` | Optional description for the dataset | `None` | | `custom_field_mapping` | `Optional[dict[str, Union[str, list[str]]]]` | Optional mapping of standard field names to custom field names in the dataset | `None` |

Returns:

| Type | Description | | --- | --- | | `Dataset` | Dataset object representing the created dataset |

Source code in `src/patronus/api/api_client.py`

```python
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

```

### upload_dataset_from_buffer_sync

```python
upload_dataset_from_buffer_sync(
    file_obj: BinaryIO,
    dataset_name: str,
    dataset_description: Optional[str] = None,
    custom_field_mapping: Optional[
        dict[str, Union[str, list[str]]]
    ] = None,
) -> api_types.Dataset

```

Upload a dataset file to create a new dataset in Patronus AI Platform.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `file_obj` | `BinaryIO` | File-like object containing dataset content (CSV or JSONL format) | *required* | | `dataset_name` | `str` | Name for the created dataset | *required* | | `dataset_description` | `Optional[str]` | Optional description for the dataset | `None` | | `custom_field_mapping` | `Optional[dict[str, Union[str, list[str]]]]` | Optional mapping of standard field names to custom field names in the dataset | `None` |

Returns:

| Type | Description | | --- | --- | | `Dataset` | Dataset object representing the created dataset |

Source code in `src/patronus/api/api_client.py`

```python
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

```

### upload_dataset_sync

```python
upload_dataset_sync(
    file_path: str,
    dataset_name: str,
    dataset_description: Optional[str] = None,
    custom_field_mapping: Optional[
        dict[str, Union[str, list[str]]]
    ] = None,
) -> api_types.Dataset

```

Upload a dataset file to create a new dataset in Patronus AI Platform.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `file_path` | `str` | Path to the dataset file (CSV or JSONL format) | *required* | | `dataset_name` | `str` | Name for the created dataset | *required* | | `dataset_description` | `Optional[str]` | Optional description for the dataset | `None` | | `custom_field_mapping` | `Optional[dict[str, Union[str, list[str]]]]` | Optional mapping of standard field names to custom field names in the dataset | `None` |

Returns:

| Type | Description | | --- | --- | | `Dataset` | Dataset object representing the created dataset |

Source code in `src/patronus/api/api_client.py`

```python
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

```

### whoami

```python
whoami() -> api_types.WhoAmIResponse

```

Fetches information about the authenticated user.

Source code in `src/patronus/api/api_client.py`

```python
async def whoami(self) -> api_types.WhoAmIResponse:
    """Fetches information about the authenticated user."""
    resp = await self.call("GET", "/v1/whoami", response_cls=api_types.WhoAmIResponse)
    resp.raise_for_status()
    return resp.data

```

### whoami_sync

```python
whoami_sync() -> api_types.WhoAmIResponse

```

Fetches information about the authenticated user.

Source code in `src/patronus/api/api_client.py`

```python
def whoami_sync(self) -> api_types.WhoAmIResponse:
    """Fetches information about the authenticated user."""
    resp = self.call_sync("GET", "/v1/whoami", response_cls=api_types.WhoAmIResponse)
    resp.raise_for_status()
    return resp.data

```

## patronus.api.api_types

### SanitizedApp

```python
SanitizedApp = Annotated[
    str,
    _create_field_sanitizer(
        "[^a-zA-Z0-9-_./ -]", max_len=50, replace_with="_"
    ),
]

```

### SanitizedLocalEvaluatorID

```python
SanitizedLocalEvaluatorID = Annotated[
    Optional[str],
    _create_field_sanitizer(
        "[^a-zA-Z0-9\\-_./]", max_len=50, replace_with="-"
    ),
]

```

### SanitizedProjectName

```python
SanitizedProjectName = Annotated[
    str, project_name_sanitizer
]

```

### project_name_sanitizer

```python
project_name_sanitizer = (
    _create_field_sanitizer(
        "[^a-zA-Z0-9_ -]", max_len=50, replace_with="_"
    ),
)

```

### Account

Bases: `BaseModel`

#### id

```python
id: str

```

#### name

```python
name: str

```

### AddEvaluatorCriteriaRevisionRequest

Bases: `BaseModel`

#### config

```python
config: dict[str, Any]

```

### AddEvaluatorCriteriaRevisionResponse

Bases: `BaseModel`

#### evaluator_criteria

```python
evaluator_criteria: EvaluatorCriteria

```

### AnnotateRequest

Bases: `BaseModel`

#### annotation_criteria_id

```python
annotation_criteria_id: str

```

#### explanation

```python
explanation: Optional[str] = None

```

#### log_id

```python
log_id: str

```

#### value_pass

```python
value_pass: Optional[bool] = None

```

#### value_score

```python
value_score: Optional[float] = None

```

#### value_text

```python
value_text: Optional[str] = None

```

### AnnotateResponse

Bases: `BaseModel`

#### evaluation

```python
evaluation: Evaluation

```

### AnnotationCategory

Bases: `BaseModel`

#### label

```python
label: Optional[str] = None

```

#### score

```python
score: Optional[float] = None

```

### AnnotationCriteria

Bases: `BaseModel`

#### annotation_type

```python
annotation_type: AnnotationType

```

#### categories

```python
categories: Optional[list[AnnotationCategory]] = None

```

#### created_at

```python
created_at: datetime

```

#### description

```python
description: Optional[str] = None

```

#### id

```python
id: str

```

#### name

```python
name: str

```

#### project_id

```python
project_id: str

```

#### updated_at

```python
updated_at: datetime

```

### AnnotationType

Bases: `str`, `Enum`

#### binary

```python
binary = 'binary'

```

#### categorical

```python
categorical = 'categorical'

```

#### continuous

```python
continuous = 'continuous'

```

#### discrete

```python
discrete = 'discrete'

```

#### text_annotation

```python
text_annotation = 'text_annotation'

```

### BatchCreateEvaluationsRequest

Bases: `BaseModel`

#### evaluations

```python
evaluations: list[ClientEvaluation] = Field(
    min_length=1, max_length=1000
)

```

### BatchCreateEvaluationsResponse

Bases: `BaseModel`

#### evaluations

```python
evaluations: list[Evaluation]

```

### ClientEvaluation

Bases: `BaseModel`

#### app

```python
app: Optional[SanitizedApp] = None

```

#### created_at

```python
created_at: Optional[datetime] = None

```

#### criteria

```python
criteria: Optional[str] = None

```

#### dataset_id

```python
dataset_id: Optional[str] = None

```

#### dataset_sample_id

```python
dataset_sample_id: Optional[str] = None

```

#### evaluation_duration

```python
evaluation_duration: Optional[timedelta] = None

```

#### evaluator_id

```python
evaluator_id: SanitizedLocalEvaluatorID

```

#### experiment_id

```python
experiment_id: Optional[str] = None

```

#### explanation

```python
explanation: Optional[str] = None

```

#### explanation_duration

```python
explanation_duration: Optional[timedelta] = None

```

#### log_id

```python
log_id: UUID

```

#### metadata

```python
metadata: Optional[dict[str, Any]] = None

```

#### metric_description

```python
metric_description: Optional[str] = None

```

#### metric_name

```python
metric_name: Optional[str] = None

```

#### pass\_

```python
pass_: Optional[bool] = Field(
    default=None, serialization_alias="pass"
)

```

#### project_id

```python
project_id: Optional[str] = None

```

#### project_name

```python
project_name: Optional[SanitizedProjectName] = None

```

#### score

```python
score: Optional[float] = None

```

#### span_id

```python
span_id: Optional[str] = None

```

#### tags

```python
tags: Optional[dict[str, str]] = None

```

#### text_output

```python
text_output: Optional[str] = None

```

#### trace_id

```python
trace_id: Optional[str] = None

```

### CreateAnnotationCriteriaRequest

Bases: `BaseModel`

#### annotation_type

```python
annotation_type: AnnotationType

```

#### categories

```python
categories: Optional[list[AnnotationCategory]] = None

```

#### description

```python
description: Optional[str] = None

```

#### name

```python
name: str = Field(min_length=1, max_length=100)

```

#### project_id

```python
project_id: str

```

### CreateAnnotationCriteriaResponse

Bases: `BaseModel`

#### annotation_criteria

```python
annotation_criteria: AnnotationCriteria

```

### CreateCriteriaRequest

Bases: `BaseModel`

#### config

```python
config: dict[str, Any]

```

#### evaluator_family

```python
evaluator_family: str

```

#### name

```python
name: str

```

### CreateCriteriaResponse

Bases: `BaseModel`

#### evaluator_criteria

```python
evaluator_criteria: EvaluatorCriteria

```

### CreateDatasetResponse

Bases: `BaseModel`

#### dataset

```python
dataset: Dataset

```

#### dataset_id

```python
dataset_id: str

```

### CreateExperimentRequest

Bases: `BaseModel`

#### metadata

```python
metadata: Optional[dict[str, Any]] = None

```

#### name

```python
name: str

```

#### project_id

```python
project_id: str

```

#### tags

```python
tags: dict[str, str] = Field(default_factory=dict)

```

### CreateExperimentResponse

Bases: `BaseModel`

#### experiment

```python
experiment: Experiment

```

### CreateProjectRequest

Bases: `BaseModel`

#### name

```python
name: SanitizedProjectName

```

### Dataset

Bases: `BaseModel`

#### created_at

```python
created_at: datetime

```

#### creation_at

```python
creation_at: Optional[datetime] = None

```

#### description

```python
description: Optional[str] = None

```

#### id

```python
id: str

```

#### name

```python
name: str

```

#### samples

```python
samples: int

```

#### type

```python
type: str

```

### DatasetDatum

Bases: `BaseModel`

#### dataset_id

```python
dataset_id: str

```

#### evaluated_model_gold_answer

```python
evaluated_model_gold_answer: Optional[str] = None

```

#### evaluated_model_input

```python
evaluated_model_input: Optional[str] = None

```

#### evaluated_model_output

```python
evaluated_model_output: Optional[str] = None

```

#### evaluated_model_retrieved_context

```python
evaluated_model_retrieved_context: Optional[list[str]] = (
    None
)

```

#### evaluated_model_system_prompt

```python
evaluated_model_system_prompt: Optional[str] = None

```

#### meta_evaluated_model_name

```python
meta_evaluated_model_name: Optional[str] = None

```

#### meta_evaluated_model_params

```python
meta_evaluated_model_params: Optional[
    dict[str, Union[str, int, float]]
] = None

```

#### meta_evaluated_model_provider

```python
meta_evaluated_model_provider: Optional[str] = None

```

#### meta_evaluated_model_selected_model

```python
meta_evaluated_model_selected_model: Optional[str] = None

```

#### sid

```python
sid: int

```

### EvaluateEvaluator

Bases: `BaseModel`

#### criteria

```python
criteria: Optional[str] = None

```

#### evaluator

```python
evaluator: str

```

#### explain_strategy

```python
explain_strategy: str = 'always'

```

### EvaluateRequest

Bases: `BaseModel`

#### app

```python
app: Optional[str] = None

```

#### capture

```python
capture: str = 'all'

```

#### dataset_id

```python
dataset_id: Optional[str] = None

```

#### dataset_sample_id

```python
dataset_sample_id: Optional[str] = None

```

#### evaluated_model_attachments

```python
evaluated_model_attachments: Optional[
    list[EvaluatedModelAttachment]
] = None

```

#### evaluated_model_gold_answer

```python
evaluated_model_gold_answer: Optional[str] = None

```

#### evaluated_model_input

```python
evaluated_model_input: Optional[str] = None

```

#### evaluated_model_output

```python
evaluated_model_output: Optional[str] = None

```

#### evaluated_model_retrieved_context

```python
evaluated_model_retrieved_context: Optional[
    Union[list[str], str]
] = None

```

#### evaluated_model_system_prompt

```python
evaluated_model_system_prompt: Optional[str] = None

```

#### evaluators

```python
evaluators: list[EvaluateEvaluator] = Field(min_length=1)

```

#### experiment_id

```python
experiment_id: Optional[str] = None

```

#### log_id

```python
log_id: Optional[str] = None

```

#### project_id

```python
project_id: Optional[str] = None

```

#### project_name

```python
project_name: Optional[str] = None

```

#### span_id

```python
span_id: Optional[str] = None

```

#### tags

```python
tags: Optional[dict[str, str]] = None

```

#### trace_id

```python
trace_id: Optional[str] = None

```

### EvaluateResponse

Bases: `BaseModel`

#### results

```python
results: list[EvaluateResult]

```

### EvaluateResult

Bases: `BaseModel`

#### criteria

```python
criteria: str

```

#### error_message

```python
error_message: Optional[str]

```

#### evaluation_result

```python
evaluation_result: Optional[EvaluationResult]

```

#### evaluator_id

```python
evaluator_id: str

```

#### status

```python
status: str

```

### EvaluatedModelAttachment

Bases: `BaseModel`

#### media_type

```python
media_type: str

```

#### url

```python
url: str

```

#### usage_type

```python
usage_type: Optional[str] = 'evaluated_model_input'

```

### Evaluation

Bases: `BaseModel`

#### annotation_criteria_id

```python
annotation_criteria_id: Optional[str] = None

```

#### app

```python
app: Optional[str] = None

```

#### created_at

```python
created_at: datetime

```

#### criteria

```python
criteria: Optional[str] = None

```

#### criteria_id

```python
criteria_id: Optional[str] = None

```

#### dataset_id

```python
dataset_id: Optional[str] = None

```

#### dataset_sample_id

```python
dataset_sample_id: Optional[str] = None

```

#### evaluation_duration

```python
evaluation_duration: Optional[timedelta] = None

```

#### evaluation_type

```python
evaluation_type: Optional[str] = None

```

#### evaluator_family

```python
evaluator_family: Optional[str] = None

```

#### evaluator_id

```python
evaluator_id: Optional[str] = None

```

#### experiment_id

```python
experiment_id: Optional[int] = None

```

#### explain_strategy

```python
explain_strategy: Optional[str] = None

```

#### explanation

```python
explanation: Optional[str] = None

```

#### explanation_duration

```python
explanation_duration: Optional[timedelta] = None

```

#### id

```python
id: int

```

#### log_id

```python
log_id: str

```

#### metadata

```python
metadata: Optional[dict[str, Any]] = None

```

#### metric_description

```python
metric_description: Optional[str] = None

```

#### metric_name

```python
metric_name: Optional[str] = None

```

#### pass\_

```python
pass_: Optional[bool] = Field(default=None, alias='pass')

```

#### project_id

```python
project_id: Optional[str] = None

```

#### score

```python
score: Optional[float] = None

```

#### span_id

```python
span_id: Optional[str] = None

```

#### tags

```python
tags: Optional[dict[str, str]] = None

```

#### text_output

```python
text_output: Optional[str] = None

```

#### trace_id

```python
trace_id: Optional[str] = None

```

#### usage

```python
usage: Optional[dict[str, Any]] = None

```

### EvaluationResult

Bases: `BaseModel`

#### additional_info

```python
additional_info: Optional[dict[str, Any]] = None

```

#### app

```python
app: Optional[str] = None

```

#### created_at

```python
created_at: Optional[AwareDatetime] = None

```

#### criteria

```python
criteria: str

```

#### dataset_id

```python
dataset_id: Optional[str] = None

```

#### dataset_sample_id

```python
dataset_sample_id: Optional[int] = None

```

#### evaluated_model_gold_answer

```python
evaluated_model_gold_answer: Optional[str] = None

```

#### evaluated_model_input

```python
evaluated_model_input: Optional[str] = None

```

#### evaluated_model_output

```python
evaluated_model_output: Optional[str] = None

```

#### evaluated_model_retrieved_context

```python
evaluated_model_retrieved_context: Optional[list[str]] = (
    None
)

```

#### evaluated_model_system_prompt

```python
evaluated_model_system_prompt: Optional[str] = None

```

#### evaluation_duration

```python
evaluation_duration: Optional[timedelta] = None

```

#### evaluation_metadata

```python
evaluation_metadata: Optional[dict] = None

```

#### evaluator_family

```python
evaluator_family: str

```

#### evaluator_id

```python
evaluator_id: str

```

#### evaluator_profile_public_id

```python
evaluator_profile_public_id: str

```

#### experiment_id

```python
experiment_id: Optional[str] = None

```

#### explanation

```python
explanation: Optional[str] = None

```

#### explanation_duration

```python
explanation_duration: Optional[timedelta] = None

```

#### id

```python
id: Optional[str] = None

```

#### pass\_

```python
pass_: Optional[bool] = Field(default=None, alias='pass')

```

#### project_id

```python
project_id: Optional[str] = None

```

#### score_raw

```python
score_raw: Optional[float] = None

```

#### tags

```python
tags: Optional[dict[str, str]] = None

```

#### text_output

```python
text_output: Optional[str] = None

```

### Evaluator

Bases: `BaseModel`

#### aliases

```python
aliases: Optional[list[str]]

```

#### default_criteria

```python
default_criteria: Optional[str] = None

```

#### evaluator_family

```python
evaluator_family: Optional[str]

```

#### id

```python
id: str

```

#### name

```python
name: str

```

### EvaluatorCriteria

Bases: `BaseModel`

#### config

```python
config: Optional[dict[str, Any]]

```

#### created_at

```python
created_at: datetime

```

#### description

```python
description: Optional[str]

```

#### evaluator_family

```python
evaluator_family: str

```

#### is_patronus_managed

```python
is_patronus_managed: bool

```

#### name

```python
name: str

```

#### public_id

```python
public_id: str

```

#### revision

```python
revision: int

```

### Experiment

Bases: `BaseModel`

#### id

```python
id: str

```

#### metadata

```python
metadata: Optional[dict[str, Any]] = None

```

#### name

```python
name: str

```

#### project_id

```python
project_id: str

```

#### tags

```python
tags: Optional[dict[str, str]] = None

```

### ExportEvaluationRequest

Bases: `BaseModel`

#### evaluation_results

```python
evaluation_results: list[ExportEvaluationResult]

```

### ExportEvaluationResponse

Bases: `BaseModel`

#### evaluation_results

```python
evaluation_results: list[ExportEvaluationResultPartial]

```

### ExportEvaluationResult

Bases: `BaseModel`

#### app

```python
app: Optional[str] = None

```

#### criteria

```python
criteria: Optional[str] = None

```

#### dataset_id

```python
dataset_id: Optional[str] = None

```

#### dataset_sample_id

```python
dataset_sample_id: Optional[int] = None

```

#### evaluated_model_attachments

```python
evaluated_model_attachments: Optional[
    list[EvaluatedModelAttachment]
] = None

```

#### evaluated_model_gold_answer

```python
evaluated_model_gold_answer: Optional[str] = None

```

#### evaluated_model_input

```python
evaluated_model_input: Optional[str] = None

```

#### evaluated_model_name

```python
evaluated_model_name: Optional[str] = None

```

#### evaluated_model_output

```python
evaluated_model_output: Optional[str] = None

```

#### evaluated_model_params

```python
evaluated_model_params: Optional[
    dict[str, Union[str, int, float]]
] = None

```

#### evaluated_model_provider

```python
evaluated_model_provider: Optional[str] = None

```

#### evaluated_model_retrieved_context

```python
evaluated_model_retrieved_context: Optional[list[str]] = (
    None
)

```

#### evaluated_model_selected_model

```python
evaluated_model_selected_model: Optional[str] = None

```

#### evaluated_model_system_prompt

```python
evaluated_model_system_prompt: Optional[str] = None

```

#### evaluation_duration

```python
evaluation_duration: Optional[timedelta] = None

```

#### evaluation_metadata

```python
evaluation_metadata: Optional[dict[str, Any]] = None

```

#### evaluator_id

```python
evaluator_id: SanitizedLocalEvaluatorID

```

#### experiment_id

```python
experiment_id: Optional[str] = None

```

#### explanation

```python
explanation: Optional[str] = None

```

#### explanation_duration

```python
explanation_duration: Optional[timedelta] = None

```

#### pass\_

```python
pass_: Optional[bool] = Field(
    default=None, serialization_alias="pass"
)

```

#### score_raw

```python
score_raw: Optional[float] = None

```

#### tags

```python
tags: Optional[dict[str, str]] = None

```

#### text_output

```python
text_output: Optional[str] = None

```

### ExportEvaluationResultPartial

Bases: `BaseModel`

#### app

```python
app: Optional[str]

```

#### created_at

```python
created_at: AwareDatetime

```

#### evaluator_id

```python
evaluator_id: str

```

#### id

```python
id: str

```

### GetAnnotationCriteriaResponse

Bases: `BaseModel`

#### annotation_criteria

```python
annotation_criteria: AnnotationCriteria

```

### GetEvaluationResponse

Bases: `BaseModel`

#### evaluation

```python
evaluation: Evaluation

```

### GetExperimentResponse

Bases: `BaseModel`

#### experiment

```python
experiment: Experiment

```

### GetProjectResponse

Bases: `BaseModel`

#### project

```python
project: Project

```

### ListAnnotationCriteriaResponse

Bases: `BaseModel`

#### annotation_criteria

```python
annotation_criteria: list[AnnotationCriteria]

```

### ListCriteriaRequest

Bases: `BaseModel`

#### evaluator_family

```python
evaluator_family: Optional[str] = None

```

#### evaluator_id

```python
evaluator_id: Optional[str] = None

```

#### get_last_revision

```python
get_last_revision: bool = False

```

#### is_patronus_managed

```python
is_patronus_managed: Optional[bool] = None

```

#### limit

```python
limit: int = 1000

```

#### name

```python
name: Optional[str] = None

```

#### offset

```python
offset: int = 0

```

#### public_id

```python
public_id: Optional[str] = None

```

#### revision

```python
revision: Optional[str] = None

```

### ListCriteriaResponse

Bases: `BaseModel`

#### evaluator_criteria

```python
evaluator_criteria: list[EvaluatorCriteria]

```

### ListDatasetData

Bases: `BaseModel`

#### data

```python
data: list[DatasetDatum]

```

### ListDatasetsResponse

Bases: `BaseModel`

#### datasets

```python
datasets: list[Dataset]

```

### ListEvaluatorsResponse

Bases: `BaseModel`

#### evaluators

```python
evaluators: list[Evaluator]

```

### Log

Bases: `BaseModel`

#### body

```python
body: Any = None

```

#### log_attributes

```python
log_attributes: Optional[dict[str, str]] = None

```

#### resource_attributes

```python
resource_attributes: Optional[dict[str, str]] = None

```

#### resource_schema_url

```python
resource_schema_url: Optional[str] = None

```

#### scope_attributes

```python
scope_attributes: Optional[dict[str, str]] = None

```

#### scope_name

```python
scope_name: Optional[str] = None

```

#### scope_schema_url

```python
scope_schema_url: Optional[str] = None

```

#### scope_version

```python
scope_version: Optional[str] = None

```

#### service_name

```python
service_name: Optional[str] = None

```

#### severity_number

```python
severity_number: Optional[int] = None

```

#### severity_test

```python
severity_test: Optional[str] = None

```

#### span_id

```python
span_id: Optional[str] = None

```

#### timestamp

```python
timestamp: Optional[datetime] = None

```

#### trace_flags

```python
trace_flags: Optional[int] = None

```

#### trace_id

```python
trace_id: Optional[str] = None

```

### Project

Bases: `BaseModel`

#### id

```python
id: str

```

#### name

```python
name: str

```

### SearchEvaluationsFilter

Bases: `BaseModel`

#### and\_

```python
and_: Optional[list[SearchEvaluationsFilter]] = None

```

#### field

```python
field: Optional[str] = None

```

#### operation

```python
operation: Optional[str] = None

```

#### or\_

```python
or_: Optional[list[SearchEvaluationsFilter]] = None

```

#### value

```python
value: Optional[Any] = None

```

### SearchEvaluationsRequest

Bases: `BaseModel`

#### filters

```python
filters: Optional[list[SearchEvaluationsFilter]] = None

```

### SearchEvaluationsResponse

Bases: `BaseModel`

#### evaluations

```python
evaluations: list[Evaluation]

```

### SearchLogsFilter

Bases: `BaseModel`

#### and\_

```python
and_: Optional[list[SearchLogsFilter]] = None

```

#### field

```python
field: Optional[str] = None

```

#### op

```python
op: Optional[str] = None

```

#### or\_

```python
or_: Optional[list[SearchLogsFilter]] = None

```

#### value

```python
value: Optional[Any] = None

```

### SearchLogsRequest

Bases: `BaseModel`

#### filters

```python
filters: Optional[list[SearchLogsFilter]] = None

```

#### limit

```python
limit: int = 1000

```

#### order

```python
order: str = 'timestamp desc'

```

### SearchLogsResponse

Bases: `BaseModel`

#### logs

```python
logs: list[Log]

```

### UpdateAnnotationCriteriaRequest

Bases: `BaseModel`

#### annotation_type

```python
annotation_type: AnnotationType

```

#### categories

```python
categories: Optional[list[AnnotationCategory]] = None

```

#### description

```python
description: Optional[str] = None

```

#### name

```python
name: str = Field(min_length=1, max_length=100)

```

### UpdateAnnotationCriteriaResponse

Bases: `BaseModel`

#### annotation_criteria

```python
annotation_criteria: AnnotationCriteria

```

### UpdateExperimentRequest

Bases: `BaseModel`

#### metadata

```python
metadata: dict[str, Any]

```

### UpdateExperimentResponse

Bases: `BaseModel`

#### experiment

```python
experiment: Experiment

```

### WhoAmIAPIKey

Bases: `BaseModel`

#### account

```python
account: Account

```

#### id

```python
id: str

```

### WhoAmICaller

Bases: `BaseModel`

#### api_key

```python
api_key: WhoAmIAPIKey

```

### WhoAmIResponse

Bases: `BaseModel`

#### caller

```python
caller: WhoAmICaller

```

### sanitize_field

```python
sanitize_field(max_length: int, sub_pattern: str)

```

Source code in `src/patronus/api/api_types.py`

```python
def sanitize_field(max_length: int, sub_pattern: str):
    def wrapper(value: str) -> str:
        if not value:
            return value
        value = value[:max_length]
        return re.sub(sub_pattern, "_", value).strip()

    return wrapper

```
