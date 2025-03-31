import datetime
import pydantic
from typing import Any, Optional
import yaml


class EvaluationResult(pydantic.BaseModel):
    """
    Container for evaluation outcomes including score, pass/fail status, explanations, and metadata.

    This class stores complete evaluation results with numeric scores, boolean pass/fail statuses,
    textual outputs, explanations, and arbitrary metadata. Evaluator functions can return instances
    of this class directly or return simpler types (bool, float, str) which will be automatically
    converted to EvaluationResult objects during recording.

    Attributes:
        score: Score of the evaluation. Can be any numerical value, though typically
            ranges from 0 to 1, where 1 represents the best possible score.
        pass_: Whether the evaluation is considered to pass or fail.
        text_output: Text output of the evaluation. Usually used for discrete
            human-readable category evaluation or as a label for score value.
        metadata: Arbitrary json-serializable metadata about evaluation.
        explanation: Human-readable explanation of the evaluation.
        tags: Key-value pair metadata.
        dataset_id: ID of the dataset associated with evaluated sample.
        dataset_sample_id: ID of the sample in a dataset associated with evaluated sample.
        evaluation_duration: Duration of the evaluation. In case value is not set,
            [@evaluator][patronus.evals.evaluators.evaluator] decorator and
            [Evaluator][patronus.evals.evaluators.Evaluator] classes will set this value automatically.
        explanation_duration: Duration of the evaluation explanation.
    """

    score: Optional[float] = None
    pass_: Optional[bool] = None
    text_output: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None
    explanation: Optional[str] = None
    tags: Optional[dict[str, str]] = None
    dataset_id: Optional[str] = None
    dataset_sample_id: Optional[str] = None
    evaluation_duration: Optional[datetime.timedelta] = None
    explanation_duration: Optional[datetime.timedelta] = None

    def format(self) -> str:
        """
        Format the evaluation result into a readable summary.
        """
        md = self.model_dump(exclude_none=True, mode="json")
        return yaml.dump(md)

    def pretty_print(self, file=None) -> None:
        """
        Pretty prints the formatted content to the specified file or standard output.
        """
        f = self.format()
        print(f, file=file)
