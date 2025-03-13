from typing import Optional
import warnings


def merge_tags(tags: Optional[dict], new_tags: Optional[dict], experiment_tags: Optional[dict]) -> dict:
    tags = tags or {}
    new_tags = new_tags or {}
    experiment_tags = experiment_tags or {}

    tags = {**tags, **new_tags}
    common_keys = set(tags.keys()) & set(experiment_tags.keys())
    diff = {key for key in common_keys if tags[key] != experiment_tags[key]}
    if diff:
        warnings.warn(f"Overriding experiment tags is not allowed. Tried to override tag: {list(diff)!r}")
    return {**tags, **experiment_tags}
