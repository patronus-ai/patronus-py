"""Pull command for downloading prompts from Patronus platform.

This module implements the `patronus prompts pull` command which downloads
prompts from the Patronus platform to local storage. It supports:

- Downloading from specific projects or all accessible projects
- Filtering prompts by name or prefix
- Selecting specific revisions or labels
- Downloading all revisions or just latest + labeled
- Verbose output and progress tracking

The command integrates with the SDK's configuration system and follows
the same authentication and project resolution patterns.
"""

from typing import Optional, Dict

import click
from rich.progress import Progress, SpinnerColumn, TextColumn

from patronus.config import config
from patronus.prompts.providers import (
    PromptProviderError,
    PromptProviderConnectionError,
    PromptProviderAuthenticationError,
)
from patronus.prompts.models import LoadedPrompt
from patronus.prompts.storage import LocalPromptStorage, PromptDefinition

from .common import (
    console,
    create_api_client,
    get_effective_resource_dir,
    get_all_projects,
    get_prompt_definitions,
    update_storage_info,
    handle_command_error,
)


@click.command()
@click.argument("prompt_name", required=False)
@click.option("--project-name", help="Target specific project")
@click.option("--all-projects", is_flag=True, help="Pull from all accessible projects")
@click.option("--name-prefix", help="Filter prompts by name prefix")
@click.option("--all-revisions", is_flag=True, help="Pull all revisions (default: latest + labeled only)")
@click.option("--revision", type=int, help="Pull specific revision")
@click.option("--label", help="Pull revisions with specific label")
@click.option("--resource-dir", help="Override resource directory (global flag)")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
@click.pass_context
def pull(
    ctx,
    prompt_name: Optional[str],
    project_name: Optional[str],
    all_projects: bool,
    name_prefix: Optional[str],
    all_revisions: bool,
    revision: Optional[int],
    label: Optional[str],
    resource_dir: Optional[str],
    verbose: bool,
) -> None:
    """Download prompts from Patronus platform to local storage.

    If neither revision nor label is specified, the latest revision is downloaded.
    You can specify a PROMPT_NAME to pull only that specific prompt.

    Examples:

    ```bash
    # Pull all prompts from Global project (latest + labeled)
    patronus prompts pull --project-name Global

    # Pull specific prompt
    patronus prompts pull my-prompt --project-name Global

    # Pull from all projects
    patronus prompts pull --all-projects

    # Pull prompts with name prefix
    patronus prompts pull --name-prefix my-agent --project-name Global

    # Pull all revisions (not just latest + labeled)
    patronus prompts pull --all-revisions --project-name Global

    # Pull specific revision
    patronus prompts pull my-prompt --revision 5 --project-name Global

    # Pull revisions with specific label
    patronus prompts pull --label production --project-name Global
    ```
    """
    try:
        # Use resource_dir from global context if not specified locally
        effective_resource_dir = get_effective_resource_dir(resource_dir, ctx.obj.get("resource_dir"))

        _pull_implementation(
            prompt_name=prompt_name,
            project_name=project_name,
            all_projects=all_projects,
            name_prefix=name_prefix,
            all_revisions=all_revisions,
            revision=revision,
            label=label,
            resource_dir=effective_resource_dir,
            verbose=verbose,
        )
    except Exception as e:
        handle_command_error(e, verbose)


def _pull_implementation(
    prompt_name: Optional[str],
    project_name: Optional[str],
    all_projects: bool,
    name_prefix: Optional[str],
    all_revisions: bool,
    revision: Optional[int],
    label: Optional[str],
    resource_dir: str,
    verbose: bool,
) -> None:
    """Implementation of the pull command."""
    if verbose:
        console.print(f"Using resource directory: {resource_dir}")

    # Create API client
    client = create_api_client(verbose)
    storage = LocalPromptStorage(resource_dir)

    # Determine target projects
    if all_projects and project_name:
        raise click.UsageError("Cannot specify both --all-projects and --project-name")

    if all_projects:
        target_projects = get_all_projects(client, verbose)
    elif project_name:
        target_projects = [project_name]
    else:
        # Use project from config
        cfg = config()
        if not cfg.project_name:
            raise click.UsageError(
                "No project specified. Use --project-name, --all-projects, or configure project_name in patronus.yaml"
            )
        target_projects = [cfg.project_name]

    if verbose:
        console.print(f"Target projects: {target_projects}")

    # Pull prompts for each project
    total_pulled = 0
    for project in target_projects:
        pulled = _pull_project_prompts(
            client=client,
            storage=storage,
            project_name=project,
            prompt_name=prompt_name,
            name_prefix=name_prefix,
            all_revisions=all_revisions,
            revision=revision,
            label=label,
            verbose=verbose,
        )
        total_pulled += pulled

    if total_pulled == 0:
        console.print("[yellow]No prompts found matching the specified criteria[/yellow]")
    else:
        console.print(f"[green]Successfully pulled {total_pulled} prompt revision(s)[/green]")


def _pull_project_prompts(
    client,
    storage: LocalPromptStorage,
    project_name: str,
    prompt_name: Optional[str],
    name_prefix: Optional[str],
    all_revisions: bool,
    revision: Optional[int],
    label: Optional[str],
    verbose: bool,
) -> int:
    """Pull prompts for a specific project."""
    try:
        if verbose:
            console.print(f"\nProcessing project: {project_name}")

        # Get prompt definitions for this project
        prompt_definitions = get_prompt_definitions(
            client=client, project_name=project_name, prompt_name=prompt_name, name_prefix=name_prefix, verbose=verbose
        )

        if not prompt_definitions:
            if verbose:
                console.print(f"No prompts found in project {project_name}")
            return 0

        total_pulled = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            disable=not verbose,
        ) as progress:
            task = progress.add_task(f"Pulling prompts from {project_name}...", total=len(prompt_definitions))

            for prompt_def in prompt_definitions:
                if verbose:
                    progress.update(task, description=f"Processing {prompt_def['name']}")

                pulled = _pull_prompt_revisions(
                    client=client,
                    storage=storage,
                    project_name=project_name,
                    prompt_def=prompt_def,
                    all_revisions=all_revisions,
                    revision=revision,
                    label=label,
                    verbose=verbose,
                )
                total_pulled += pulled

                if verbose:
                    progress.advance(task)

        return total_pulled

    except Exception as e:
        if isinstance(e, (PromptProviderError, PromptProviderConnectionError, PromptProviderAuthenticationError)):
            raise
        raise PromptProviderError(f"Failed to pull prompts from project {project_name}: {e}")


def _pull_prompt_revisions(
    client,
    storage: LocalPromptStorage,
    project_name: str,
    prompt_def: Dict,
    all_revisions: bool,
    revision: Optional[int],
    label: Optional[str],
    verbose: bool,
) -> int:
    """Pull revisions for a specific prompt."""
    try:
        # Get revisions
        if revision is not None:
            # Get specific revision
            resp = client.prompts.list_revisions(
                prompt_name=prompt_def["name"], project_name=project_name, revision=revision
            )
        elif label is not None:
            # Get revisions with specific label
            resp = client.prompts.list_revisions(prompt_name=prompt_def["name"], project_name=project_name, label=label)
        elif all_revisions:
            # Get all revisions
            resp = client.prompts.list_revisions(prompt_name=prompt_def["name"], project_name=project_name)
        else:
            # Get latest + labeled revisions (default behavior)
            resp = client.prompts.list_revisions(prompt_name=prompt_def["name"], project_name=project_name)

        if not resp.prompt_revisions:
            if verbose:
                console.print(f"No revisions found for prompt {prompt_def['name']}")
            return 0

        # Filter revisions based on criteria
        revisions_to_pull = []

        if all_revisions or revision is not None or label is not None:
            # Pull all returned revisions
            revisions_to_pull = resp.prompt_revisions
        else:
            # Default: latest + labeled revisions
            latest_revision = max(resp.prompt_revisions, key=lambda r: r.revision)
            revisions_to_pull.append(latest_revision)

            # Add any labeled revisions that aren't the latest
            for rev in resp.prompt_revisions:
                if rev.labels and rev.revision != latest_revision.revision:
                    revisions_to_pull.append(rev)

        # Remove duplicates
        unique_revisions = {}
        for rev in revisions_to_pull:
            unique_revisions[rev.revision] = rev
        revisions_to_pull = list(unique_revisions.values())

        # Save prompt definition
        definition = PromptDefinition(
            prompt_definition_id=prompt_def["id"],
            project_id=prompt_def["project_id"],
            name=prompt_def["name"],
            latest_revision=max(r.revision for r in revisions_to_pull),
        )
        storage.save_prompt_definition(project_name, prompt_def["name"], definition)

        # Save description
        storage.save_prompt_description(project_name, prompt_def["name"], prompt_def["description"])

        # Save each revision
        for rev in revisions_to_pull:
            loaded_prompt = LoadedPrompt(
                prompt_definition_id=prompt_def["id"],
                project_id=prompt_def["project_id"],
                project_name=project_name,
                name=prompt_def["name"],
                description=prompt_def["description"],
                revision_id=rev.id,
                revision=rev.revision,
                body=rev.body,
                normalized_body_sha256=rev.normalized_body_sha256,
                metadata=rev.metadata,
                labels=rev.labels,
                created_at=rev.created_at,
            )
            storage.save_prompt_revision(project_name, prompt_def["name"], loaded_prompt)

            if verbose:
                labels_str = ", ".join(rev.labels) if rev.labels else "none"
                console.print(f"  Saved revision {rev.revision} (labels: {labels_str})")

        # Update storage info with project and prompt information
        update_storage_info(storage, project_name, prompt_def["project_id"], prompt_def["name"], revisions_to_pull)

        return len(revisions_to_pull)

    except Exception as e:
        if isinstance(e, (PromptProviderError, PromptProviderConnectionError, PromptProviderAuthenticationError)):
            raise
        raise PromptProviderError(f"Failed to pull revisions for prompt {prompt_def['name']}: {e}")
