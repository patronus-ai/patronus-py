from enum import Enum


class EventNames(str, Enum):
    evaluation_data = "pat.evaluation.data"


class Attributes(str, Enum):
    event_name = "event.name"
    log_id = "pat.log.id"
    log_type = "pat.log.type"
    span_type = "pat.span.type"
    project_name = "pat.project.name"
    app = "pat.app"
    experiment_id = "pat.experiment.id"
    experiment_name = "pat.experiment.name"
    evaluator_id = "pat.evaluator.id"
    evaluator_criteria = "pat.evaluator.criteria"


class GenAIAttributes(str, Enum):
    operation_name = "gen_ai.operation.name"


class OperationNames(str, Enum):
    eval = "evaluation"
    task = "task"


class LogTypes(str, Enum):
    # Eval log type is emitted by evaluators, they contain evaluation data
    eval = "eval"
    # Trace log type is emitted by traced() decorator.
    trace = "trace"
    # User log type is emitted any time user uses the logger directly.
    user = "user"


class SpanTypes(str, Enum):
    eval = "eval"
    experiment_sample = "experiment.sample"
    experiment_chain_step = "experiment.chain.step"
    experiment_task = "experiment.task"
