import datetime
import pydantic
import textwrap
from typing import Any, Optional


class EvaluationResult(pydantic.BaseModel):
    score: Optional[float] = pydantic.Field(
        default=None,
        description=textwrap.dedent(
            """\
        Score of the evaluation. Can be any numerical value, though typically ranges from 0 to 1,
        where 1 represents the best possible score.
        """
        ),
    )
    # Whether the evaluation is considered to pass or fail
    pass_: Optional[bool] = None
    # Text output of the evaluation.
    # Usually used for discrete human-readable category evaluation or as a label for score value.
    text_output: Optional[str] = None
    # Arbitrary json-serializable metadata about evaluation.
    metadata: Optional[dict[str, Any]] = None
    # Human-readable explanation of the evaluation.
    explanation: Optional[str] = None
    # Key-value pair metadata
    tags: Optional[dict[str, str]] = None
    # ID of the dataset associated with evaluated sample
    dataset_id: Optional[str] = None
    # ID of the sample in a dataset associated with evaluated sample
    dataset_sample_id: Optional[str] = None
    # Duration of the evaluation.
    # In case value is not set, @evaluator decorator and Evaluator classes will set this value automatically.
    evaluation_duration: Optional[datetime.timedelta] = None
    # Duration of the evaluation explanation.
    explanation_duration: Optional[datetime.timedelta] = None
