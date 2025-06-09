# CLI Overview

The Patronus CLI provides command-line tools for managing prompts and interacting with the Patronus platform. It integrates seamlessly with the existing Patronus SDK configuration and storage, enabling offline/local prompt management workflows.

## Features

The CLI currently supports prompt management with plans for future expansion:

### Prompt Management
- **Download prompts** from the Patronus platform to local storage
- **Clean up local storage** by removing outdated prompt revisions
- **Project-based organization** with support for multiple projects
- **Version control-like experience** for prompt management
- **Offline workflows** with local prompt storage

### Future Features
- Prompt upload capabilities (`patronus prompts push`)
- Prompt synchronization (`patronus prompts sync`)
- Interactive command modes
- Enhanced output formats

## Installation

The CLI is included with the Patronus Python SDK:

```bash
pip install patronus
```

Once installed, the CLI is available as the `patronus` command:

```bash
patronus --help
```

## Configuration

The CLI uses the same configuration system as the Patronus SDK, supporting multiple configuration sources with the following precedence:

1. **Command-line arguments** (highest priority)
2. **Environment variables** (`PATRONUS_*`)
3. **Configuration file** (`patronus.yaml`)
4. **Default values** (lowest priority)

### Environment Variables

```bash
export PATRONUS_API_KEY="your-api-key"
export PATRONUS_PROJECT_NAME="your-project"
export PATRONUS_RESOURCE_DIR=".patronus"
```

### Configuration File

Create a `patronus.yaml` file in your working directory:

```yaml
api_key: "your-api-key"
project_name: "your-project"
resource_dir: ".patronus"
```

### Authentication

Authentication is handled automatically using your Patronus API key. You can verify your authentication status with:

```bash
patronus whoami
```

## Local Storage

The CLI manages prompts in a local directory structure (default: `.patronus/`):

```
.patronus/
├── info.yaml                    # Project and prompt metadata
└── <project_name>/
    └── prompts/
        └── <prompt_name>/
            ├── description.txt   # Prompt description
            ├── v1/
            │   ├── prompt.txt    # Prompt body
            │   └── metadata.json # Revision metadata
            ├── v2/
            └── v4/              # Latest revision
                ├── prompt.txt
                └── metadata.json
```

### Integration with SDK

The local storage integrates seamlessly with the SDK's prompt loading system. Once prompts are downloaded via the CLI, they can be loaded in your Python code:

```python
import patronus

patronus.init()

# SDK will automatically use locally stored prompts when available
prompt = patronus.prompts.get("my-prompt")
```

## Global Options

All CLI commands support these global options:

- `--resource-dir PATH` - Override the default resource directory
- `--help` - Show help information
- `--version` - Show version information

## Error Handling

The CLI provides clear, actionable error messages and uses standard exit codes:

- `0` - Success
- `1` - General error
- `2` - Configuration error
- `3` - Authentication error
- `4` - Resource not found
- `5` - File system error

Use the `-v/--verbose` flag with any command to get detailed error information and troubleshooting guidance.

## Examples

### Basic Workflow

```bash
# Verify authentication
patronus whoami

# Download all prompts from a specific project
patronus prompts pull --project-name "My Project"

# Download a specific prompt
patronus prompts pull my-prompt --project-name "My Project"

# Clean up old revisions
patronus prompts tidy

# View detailed help for any command
patronus prompts pull --help
```

### Advanced Usage

```bash
# Download from all accessible projects
patronus prompts pull --all-projects

# Download prompts with a specific prefix
patronus prompts pull --name-prefix "agent-" --project-name "AI Agents"

# Download all revisions (not just latest + labeled)
patronus prompts pull --all-revisions --project-name "Development"

# Use custom resource directory
patronus prompts pull --resource-dir ./my-prompts --project-name "Global"
```

## Getting Help

- Use `patronus --help` for general CLI help
- Use `patronus <command> --help` for command-specific help
- Add `-v/--verbose` to any command for detailed output
- Check the [Configuration Guide](../configuration.md) for setup assistance
