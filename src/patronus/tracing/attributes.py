from enum import Enum
from typing import Optional


class Attributes(str, Enum):
    log_type = "pat.log_type"
    project_name = "pat.project.name"
    app = "pat.app"
    experiment_id = "pat.experiment.id"
    log_id = "pat.log_id"


def format_service_name(project_name: str, app: Optional[str] = None, experiment_id: Optional[str] = None) -> str:
    service_name = f"{project_name}/"
    if experiment_id:
        service_name += f"ex:{experiment_id}"
    else:
        service_name += f"app:{app}"
    return service_name
