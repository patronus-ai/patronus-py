"""Models for local storage of prompts."""

import dataclasses
import datetime


@dataclasses.dataclass
class LabelInfo:
    """Information about a label pointing to a specific revision."""

    name: str
    points_to: "RevisionPointer"


@dataclasses.dataclass
class RevisionPointer:
    """Points to a specific revision version."""

    version: str  # e.g., "v4"


@dataclasses.dataclass
class ProjectInfo:
    """Information about a project in local storage."""

    name: str
    id: str


@dataclasses.dataclass
class PromptInfo:
    """Information about a prompt in local storage."""

    name: str
    project_name: str
    labels: list[LabelInfo]


@dataclasses.dataclass
class StorageInfo:
    """Top-level storage information."""

    projects: list[ProjectInfo]
    prompts: list[PromptInfo]


@dataclasses.dataclass
class PromptDefinition:
    """Prompt definition metadata."""

    prompt_definition_id: str
    project_id: str
    name: str
    latest_revision: int


@dataclasses.dataclass
class RevisionMetadata:
    """System metadata for a prompt revision."""

    revision_id: str
    revision: int
    normalized_body_sha256: str
    labels: list[str]
    created_at: datetime.datetime
