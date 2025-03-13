# Configuration

The Patronus Experimentation Framework offers several configuration options that can be set in the following ways:

1. In Code
2. Environment Variables
3. YAML Configuration File

Configuration options are prioritized in the order listed above.

| Config name   | Environment Variable   | Default Value                   |
|---------------|------------------------|---------------------------------|
| project_name  | PATRONUS_PROJECT_NAE   | `Global`                        |
| app           | PATRONUS_APP           | `default`                       |
| api_key       | PATRONUS_API_KEY       |                                 |
| api_url       | PATRONUS_API_URL       | `https://api.patronus.ai`       |
| ui_url        | PATRONUS_UI_URL        | `https://app.patronus.ai`       |
| otel_endpoint | PATRONUS_OTEL_ENDPOINT | `https://otel.patronus.ai:4317` |
| timeout_s     | PATRONUS_TIMEOUT_S     | `300`                           |

## Configuration File (`patronus.yaml`)

You can also provide configuration options using a patronus.yaml file. This file must be present in the working
directory when executing your script.

```yaml
project_name: "my-project"
app: "my-agent"

api_key: "YOUR_API_KEY"
api_url: "https://api.patronus.ai"
ui_url: "https://app.patronus.ai"

```
