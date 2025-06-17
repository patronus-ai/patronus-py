# Configuration

The Patronus Experimentation Framework offers several configuration options that can be set in the following ways:

1. Through function parameters (in code)
2. Environment variables
3. YAML configuration file

Configuration options are prioritized in the order listed above, meaning that if a configuration value is provided through function parameters, it will override values from environment variables or the YAML file.

## Configuration Options

| Config name                 | Environment Variable                 | Default Value                                                                      |
|-----------------------------|--------------------------------------|------------------------------------------------------------------------------------|
| service                     | PATRONUS_SERVICE                     | Defaults to value retrieved from `OTEL_SERVICE_NAME` env var or `platform.node()`. |
| project_name                | PATRONUS_PROJECT_NAME                | `Global`                                                                           |
| app                         | PATRONUS_APP                         | `default`                                                                          |
| api_key                     | PATRONUS_API_KEY                     |                                                                                    |
| api_url                     | PATRONUS_API_URL                     | `https://api.patronus.ai`                                                          |
| ui_url                      | PATRONUS_UI_URL                      | `https://app.patronus.ai`                                                          |
| otel_endpoint               | PATRONUS_OTEL_ENDPOINT               | `https://otel.patronus.ai:4317`                                                    |
| otel_exporter_otlp_protocol | PATRONUS_OTEL_EXPORTER_OTLP_PROTOCOL | Falls back to OTEL env vars, defaults to `grpc`                                    |
| timeout_s                   | PATRONUS_TIMEOUT_S                   | `300`                                                                              |
| prompt_templating_engine    | PATRONUS_PROMPT_TEMPLATING_ENGINE    | `f-string`                                                                         |
| prompt_providers            | PATRONUS_PROMPT_PROVIDERS            | `["local", "api"]`                                                                 |
| resource_dir                | PATRONUS_RESOURCE_DIR                | `./patronus`                                                                       |

## Configuration Methods

### 1. Function Parameters

You can provide configuration options directly through function parameters when calling key Patronus functions.

#### Using [`init()`][patronus.init.init]

Use the `init()` function when you need to set up the Patronus SDK for evaluations, logging, and tracing outside of experiments. This initializes the global context used by the SDK.

```python
import patronus

# Initialize with specific configuration
patronus.init(
    project_name="my-project",
    app="recommendation-service",
    api_key="your-api-key",
    api_url="https://api.patronus.ai",
    service="my-service",
    prompt_templating_engine="mustache"
)
```

#### Using [`run_experiment()`][patronus.experiments.experiment.run_experiment] or [`Experiment.create()`][patronus.experiments.experiment.Experiment.create]

Use these functions when running experiments. They handle their own initialization, so you don't need to call `init()` separately. Experiments create their own context scoped to the experiment.

```python
from patronus import run_experiment

# Run experiment with specific configuration
experiment = run_experiment(
    dataset=my_dataset,
    task=my_task,
    evaluators=[my_evaluator],
    project_name="my-project",
    api_key="your-api-key",
    service="my-service"
)
```

### 2. Environment Variables

You can set configuration options using environment variables with the prefix `PATRONUS_`:

```bash
export PATRONUS_API_KEY="your-api-key"
export PATRONUS_PROJECT_NAME="my-project"
export PATRONUS_SERVICE="my-service"
```

### 3. YAML Configuration File (`patronus.yaml`)

You can also provide configuration options using a `patronus.yaml` file. This file must be present in the working
directory when executing your script.

```yaml
service: "my-service"
project_name: "my-project"
app: "my-agent"

api_key: "YOUR_API_KEY"
api_url: "https://api.patronus.ai"
ui_url: "https://app.patronus.ai"
otel_endpoint: "https://otel.patronus.ai:4317"
otel_exporter_otlp_protocol: "grpc"  # or "http/protobuf"
timeout_s: 300

# Prompt management configuration
prompt_templating_engine: "mustache"
prompt_providers: [ "local", "api" ]
resource_dir: "./my-resources"
```

## Configuration Precedence

When determining the value for a configuration option, Patronus follows this order of precedence:

1. Function parameter values (highest priority)
2. Environment variables
3. YAML configuration file
4. Default values (lowest priority)

For example, if you provide `project_name` as a function parameter and also have it defined in your environment variables and YAML file, the function parameter value will be used.

## Programmatic Configuration Access

For more advanced use cases, you can directly access the configuration system through the [`Config`][patronus.config.Config] class and the [`config()`][patronus.config.config] function:

```python
from patronus.config import config

# Access the configuration singleton
cfg = config()

# Read configuration values
api_key = cfg.api_key
project_name = cfg.project_name

# Check for specific conditions
if cfg.api_url != "https://api.patronus.ai":
    print("Using custom API endpoint")
```

This approach is particularly useful when you need to inspect or log the current configuration state.

## Observability Configuration

For detailed information about configuring observability features like tracing and logging, including exporter protocol selection and endpoint configuration, see the [Observability Configuration](observability/configuration.md) guide.
