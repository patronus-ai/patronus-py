from enum import Enum
from typing import Optional


class Attributes(str, Enum):
    log_id = "pat.log.id"
    log_type = "pat.log.type"
    project_name = "pat.project.name"
    app = "pat.app"
    experiment_id = "pat.experiment.id"
    evaluator_id = "pat.evaluator.id"
    evaluator_criteria = "pat.evaluator.criteria"


class GenAIAttributes(str, Enum):
    operation_name = "gen_ai.operation.name"


class OperationNames(str, Enum):
    eval = "eval"
    task = "task"


class LogTypes(str, Enum):
    # Eval log type is emitted by evaluators, they contain evaluation data
    eval = "eval"
    # Trace log type is emitted by traced() decorator.
    trace = "trace"
    # User log type is emitted any time user uses the logger directly.
    user = "user"


def format_service_name(project_name: str, app: Optional[str] = None, experiment_id: Optional[str] = None) -> str:
    service_name = f"{project_name}/"
    if experiment_id:
        service_name += f"ex:{experiment_id}"
    else:
        service_name += f"app:{app}"
    return service_name
