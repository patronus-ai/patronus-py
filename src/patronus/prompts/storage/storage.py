"""Local prompt storage system."""

import json
import datetime
from pathlib import Path
from typing import Optional

import yaml

from .models import StorageInfo, ProjectInfo, PromptInfo, LabelInfo, RevisionPointer, PromptDefinition, RevisionMetadata
from patronus.prompts.models import LoadedPrompt


class LocalPromptStorage:
    """Manages local storage of prompts and metadata."""

    def __init__(self, resource_dir: str):
        self.resource_dir = Path(resource_dir)
        self.info_file = self.resource_dir / "info.yaml"

    def ensure_resource_dir(self) -> None:
        """Ensure the resource directory exists."""
        self.resource_dir.mkdir(parents=True, exist_ok=True)

    def load_storage_info(self) -> StorageInfo:
        """Load storage info from info.yaml, return empty if file doesn't exist."""
        if not self.info_file.exists():
            return StorageInfo(projects=[], prompts=[])

        try:
            with open(self.info_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            # Convert projects
            projects = []
            for p in data.get("projects", []):
                projects.append(ProjectInfo(name=p["name"], id=p["id"]))

            # Convert prompts
            prompts = []
            for p in data.get("prompts", []):
                labels = []
                for label in p.get("labels", []):
                    labels.append(
                        LabelInfo(name=label["name"], points_to=RevisionPointer(version=label["points_to"]["version"]))
                    )
                prompts.append(PromptInfo(name=p["name"], project_name=p["project_name"], labels=labels))

            return StorageInfo(projects=projects, prompts=prompts)
        except Exception as e:
            raise RuntimeError(f"Failed to load storage info from {self.info_file}: {e}")

    def save_storage_info(self, storage_info: StorageInfo) -> None:
        """Save storage info to info.yaml."""
        self.ensure_resource_dir()

        # Convert to dict format
        data = {
            "projects": [{"name": p.name, "id": p.id} for p in storage_info.projects],
            "prompts": [
                {
                    "name": p.name,
                    "project_name": p.project_name,
                    "labels": [
                        {"name": label.name, "points_to": {"version": label.points_to.version}} for label in p.labels
                    ],
                }
                for p in storage_info.prompts
            ],
        }

        # Add header comment
        header = (
            "# This file is automatically managed by Patronus CLI\n"
            "# Do not edit manually as changes will be overwritten\n\n"
        )

        try:
            with open(self.info_file, "w", encoding="utf-8") as f:
                f.write(header)
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            raise RuntimeError(f"Failed to save storage info to {self.info_file}: {e}")

    def get_prompt_dir(self, project_name: str, prompt_name: str) -> Path:
        """Get the directory path for a specific prompt."""
        return self.resource_dir / project_name / "prompts" / prompt_name

    def get_revision_dir(self, project_name: str, prompt_name: str, revision: int) -> Path:
        """Get the directory path for a specific revision."""
        prompt_dir = self.get_prompt_dir(project_name, prompt_name)
        return prompt_dir / "revisions" / f"v{revision}"

    def save_prompt_definition(self, project_name: str, prompt_name: str, definition: PromptDefinition) -> None:
        """Save prompt definition metadata."""
        prompt_dir = self.get_prompt_dir(project_name, prompt_name)
        prompt_dir.mkdir(parents=True, exist_ok=True)

        definition_file = prompt_dir / "definition.json"
        definition_data = {
            "prompt_definition_id": definition.prompt_definition_id,
            "project_id": definition.project_id,
            "name": definition.name,
            "latest_revision": definition.latest_revision,
        }

        try:
            with open(definition_file, "w", encoding="utf-8") as f:
                json.dump(definition_data, f, indent=2)
        except Exception as e:
            raise RuntimeError(f"Failed to save prompt definition to {definition_file}: {e}")

    def load_prompt_definition(self, project_name: str, prompt_name: str) -> Optional[PromptDefinition]:
        """Load prompt definition metadata."""
        definition_file = self.get_prompt_dir(project_name, prompt_name) / "definition.json"
        if not definition_file.exists():
            return None

        try:
            with open(definition_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            return PromptDefinition(
                prompt_definition_id=data["prompt_definition_id"],
                project_id=data["project_id"],
                name=data["name"],
                latest_revision=data["latest_revision"],
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load prompt definition from {definition_file}: {e}")

    def save_prompt_description(self, project_name: str, prompt_name: str, description: Optional[str]) -> None:
        """Save prompt description."""
        if description is None:
            return

        prompt_dir = self.get_prompt_dir(project_name, prompt_name)
        prompt_dir.mkdir(parents=True, exist_ok=True)

        description_file = prompt_dir / "description.txt"
        try:
            with open(description_file, "w", encoding="utf-8") as f:
                f.write(description)
        except Exception as e:
            raise RuntimeError(f"Failed to save description to {description_file}: {e}")

    def load_prompt_description(self, project_name: str, prompt_name: str) -> Optional[str]:
        """Load prompt description."""
        description_file = self.get_prompt_dir(project_name, prompt_name) / "description.txt"
        if not description_file.exists():
            return None

        try:
            with open(description_file, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            raise RuntimeError(f"Failed to load description from {description_file}: {e}")

    def save_prompt_revision(self, project_name: str, prompt_name: str, loaded_prompt: LoadedPrompt) -> None:
        """Save a complete prompt revision including body, metadata, and system metadata."""
        revision_dir = self.get_revision_dir(project_name, prompt_name, loaded_prompt.revision)
        revision_dir.mkdir(parents=True, exist_ok=True)

        # Save prompt body
        prompt_file = revision_dir / "prompt.txt"
        try:
            with open(prompt_file, "w", encoding="utf-8") as f:
                f.write(loaded_prompt.body)
        except Exception as e:
            raise RuntimeError(f"Failed to save prompt body to {prompt_file}: {e}")

        # Save user metadata
        if loaded_prompt.metadata:
            metadata_file = revision_dir / "metadata.json"
            try:
                with open(metadata_file, "w", encoding="utf-8") as f:
                    json.dump(loaded_prompt.metadata, f, indent=2)
            except Exception as e:
                raise RuntimeError(f"Failed to save metadata to {metadata_file}: {e}")

        # Save system metadata
        revision_metadata = RevisionMetadata(
            revision_id=loaded_prompt.revision_id,
            revision=loaded_prompt.revision,
            normalized_body_sha256=loaded_prompt.normalized_body_sha256,
            labels=loaded_prompt.labels,
            created_at=loaded_prompt.created_at,
        )
        self.save_revision_metadata(project_name, prompt_name, loaded_prompt.revision, revision_metadata)

    def save_revision_metadata(
        self, project_name: str, prompt_name: str, revision: int, metadata: RevisionMetadata
    ) -> None:
        """Save system metadata for a revision."""
        revision_dir = self.get_revision_dir(project_name, prompt_name, revision)
        revision_dir.mkdir(parents=True, exist_ok=True)

        revision_file = revision_dir / "revision.json"
        revision_data = {
            "revision_id": metadata.revision_id,
            "revision": metadata.revision,
            "normalized_body_sha256": metadata.normalized_body_sha256,
            "labels": metadata.labels,
            "created_at": metadata.created_at.isoformat(),
        }

        try:
            with open(revision_file, "w", encoding="utf-8") as f:
                json.dump(revision_data, f, indent=2)
        except Exception as e:
            raise RuntimeError(f"Failed to save revision metadata to {revision_file}: {e}")

    def load_revision_metadata(self, project_name: str, prompt_name: str, revision: int) -> Optional[RevisionMetadata]:
        """Load system metadata for a revision."""
        revision_file = self.get_revision_dir(project_name, prompt_name, revision) / "revision.json"
        if not revision_file.exists():
            return None

        try:
            with open(revision_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            return RevisionMetadata(
                revision_id=data["revision_id"],
                revision=data["revision"],
                normalized_body_sha256=data["normalized_body_sha256"],
                labels=data["labels"],
                created_at=datetime.datetime.fromisoformat(data["created_at"]),
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load revision metadata from {revision_file}: {e}")

    def load_prompt_revision(self, project_name: str, prompt_name: str, revision: int) -> Optional[LoadedPrompt]:
        """Load a complete prompt revision."""
        revision_dir = self.get_revision_dir(project_name, prompt_name, revision)
        if not revision_dir.exists():
            return None

        # Load definition
        definition = self.load_prompt_definition(project_name, prompt_name)
        if not definition:
            return None

        # Load description
        description = self.load_prompt_description(project_name, prompt_name)

        # Load prompt body
        prompt_file = revision_dir / "prompt.txt"
        if not prompt_file.exists():
            return None

        try:
            with open(prompt_file, "r", encoding="utf-8") as f:
                body = f.read()
        except Exception as e:
            raise RuntimeError(f"Failed to load prompt body from {prompt_file}: {e}")

        # Load user metadata
        metadata_file = revision_dir / "metadata.json"
        metadata = None
        if metadata_file.exists():
            try:
                with open(metadata_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            except Exception as e:
                raise RuntimeError(f"Failed to load metadata from {metadata_file}: {e}")

        # Load system metadata
        revision_metadata = self.load_revision_metadata(project_name, prompt_name, revision)
        if not revision_metadata:
            return None

        return LoadedPrompt(
            prompt_definition_id=definition.prompt_definition_id,
            project_id=definition.project_id,
            project_name=project_name,
            name=prompt_name,
            description=description,
            revision_id=revision_metadata.revision_id,
            revision=revision_metadata.revision,
            body=body,
            normalized_body_sha256=revision_metadata.normalized_body_sha256,
            metadata=metadata,
            labels=revision_metadata.labels,
            created_at=revision_metadata.created_at,
        )

    def list_available_revisions(self, project_name: str, prompt_name: str) -> list[int]:
        """List all available revisions for a prompt."""
        prompt_dir = self.get_prompt_dir(project_name, prompt_name)
        revisions_dir = prompt_dir / "revisions"

        if not revisions_dir.exists():
            return []

        revisions = []
        for item in revisions_dir.iterdir():
            if item.is_dir() and item.name.startswith("v"):
                try:
                    revision = int(item.name[1:])  # Remove 'v' prefix
                    revisions.append(revision)
                except ValueError:
                    continue  # Skip invalid directory names

        return sorted(revisions)

    def prompt_exists(self, project_name: str, prompt_name: str) -> bool:
        """Check if a prompt exists in local storage."""
        prompt_dir = self.get_prompt_dir(project_name, prompt_name)
        return prompt_dir.exists() and (prompt_dir / "definition.json").exists()

    def list_prompts(self, project_name: Optional[str] = None) -> list[tuple[str, str]]:
        """List all prompts in storage, optionally filtered by project.

        Returns:
            List of (project_name, prompt_name) tuples
        """
        storage_info = self.load_storage_info()
        results = []

        for prompt in storage_info.prompts:
            if project_name is None or prompt.project_name == project_name:
                results.append((prompt.project_name, prompt.name))

        return results

    def get_projects(self) -> list[ProjectInfo]:
        """Get all projects in storage."""
        storage_info = self.load_storage_info()
        return storage_info.projects

    def update_project_info(self, project_name: str, project_id: str) -> None:
        """Update or add project information."""
        storage_info = self.load_storage_info()

        # Check if project exists
        for project in storage_info.projects:
            if project.name == project_name:
                if project.id != project_id:
                    raise RuntimeError(
                        f"Project ID mismatch for {project_name}. Expected {project.id}, got {project_id}"
                    )
                return

        # Add new project
        storage_info.projects.append(ProjectInfo(name=project_name, id=project_id))
        self.save_storage_info(storage_info)

    def update_prompt_labels(self, project_name: str, prompt_name: str, labels: list[tuple[str, int]]) -> None:
        """Update label information for a prompt.

        Args:
            project_name: Name of the project
            prompt_name: Name of the prompt
            labels: List of (label_name, revision) tuples
        """
        storage_info = self.load_storage_info()

        # Find or create prompt info
        prompt_info = None
        for prompt in storage_info.prompts:
            if prompt.name == prompt_name and prompt.project_name == project_name:
                prompt_info = prompt
                break

        if prompt_info is None:
            prompt_info = PromptInfo(name=prompt_name, project_name=project_name, labels=[])
            storage_info.prompts.append(prompt_info)

        # Update labels
        prompt_info.labels.clear()
        for label_name, revision in labels:
            prompt_info.labels.append(LabelInfo(name=label_name, points_to=RevisionPointer(version=f"v{revision}")))

        self.save_storage_info(storage_info)
