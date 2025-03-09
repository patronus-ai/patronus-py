import types
import contextlib
import typing
from contextvars import ContextVar


class EvaluationAttributesContext(typing.TypedDict):
    tags: typing.Optional[dict[str, str]]
    experiment_tags: typing.Optional[dict[str, str]]
    dataset_id: typing.Optional[str]
    dataset_sample_id: typing.Optional[str]


_empty_evaluation_attributes = types.MappingProxyType(
    {
        "tags": None,
        "experiment_tags": None,
        "dataset_id": None,
        "dataset_sample_id": None,
    }
)

_ctx_evaluation_attributes: ContextVar[EvaluationAttributesContext] = ContextVar("_ctx_evaluation_attributes")


@contextlib.contextmanager
def evaluation_attributes(attrs: EvaluationAttributesContext):
    token = _ctx_evaluation_attributes.set(attrs)
    yield
    _ctx_evaluation_attributes.reset(token)


def get_context_evaluation_attributes() -> EvaluationAttributesContext:
    return _ctx_evaluation_attributes.get(_empty_evaluation_attributes)
