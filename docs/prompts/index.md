# Prompt Management

The Patronus SDK provides tools to version, retrieve, and render prompts in your LLM applications.

## Quick Start

### Creating a Prompt

```python
import patronus
import textwrap
from patronus.prompts import Prompt, push_prompt

patronus.init()

# Create a new prompt
prompt = Prompt(
    name="support/troubleshooting/login-issues",
    body=textwrap.dedent("""
        You are a support specialist for {product_name}.
        ISSUE: {issue_description}
        TIER: {subscription_tier}

        Provide a solution for this {issue_type} problem. Be concise.
        Include steps and end with an offer for further help.
        """),
    description="Support prompt for login issues",
    metadata={"temperature": 0.7, "tone": "helpful"}
)

# Push the prompt to Patronus
loaded_prompt = push_prompt(prompt)

# Render the prompt
rendered = prompt.render(
    issue_description="Cannot log in with correct credentials",
    product_name="CloudWorks",
    subscription_tier="Business",
    issue_type="authentication"
)
print(rendered)
```

### Loading a Prompt

```python
import patronus
from patronus.prompts import load_prompt

patronus.init()

# Get the latest version of the prompt we just created
prompt = load_prompt(name="support/troubleshooting/login-issues")

# Access metadata
print(prompt.metadata)

# Render the prompt with different parameters
rendered = prompt.render(
    issue_description="Password reset link not working",
    product_name="CloudWorks",
    subscription_tier="Enterprise",
    issue_type="password reset"
)
print(rendered)
```

## Loading Prompts

Use `load_prompt` to retrieve prompts from the Patronus platform:

```python
import patronus
from patronus.prompts import load_prompt

patronus.init()

# Load an instruction prompt that doesn't need any parameters
prompt = load_prompt(name="content/writing/blog-instructions")
rendered = prompt.render()
print(rendered)
```

For async applications:

```python
from patronus.prompts import aload_prompt

prompt = await aload_prompt(name="content/writing/blog-instructions")
```

### Loading Specific Versions

Retrieve prompts by revision number or label:

```python
# Load a specific revision
prompt = load_prompt(name="content/blog/technical-explainer", revision=3)

# Load by label (production environment)
prompt = load_prompt(name="legal/contracts/privacy-policy", label="production")
```

## Creating and Updating Prompts

Create new prompts using `push_prompt`:

```python
from patronus.prompts import Prompt, push_prompt

new_prompt = Prompt(
    name="dev/bug-fix/python-error",
    body="Fix this Python code error: {error_message}. Code: ```python\n{code_snippet}\n```",
    description="Template for Python debugging assistance",
    metadata={
        "creator": "dev-team",
        "temperature": 0.7,
        "max_tokens": 250
    }
)

loaded_prompt = push_prompt(new_prompt)
```

For async applications:

```python
from patronus.prompts import apush_prompt

loaded_prompt = await apush_prompt(new_prompt)
```

The `push_prompt` function automatically handles duplicate detection - if a prompt with identical content already exists, it returns the existing revision instead of creating a new one.

## Rendering Prompts

Render prompts with variables:

```python
rendered = prompt.render(user_query="How do I optimize database performance?", expertise_level="intermediate")
```

### Template Engines

Patronus supports multiple template engines:

```python
# F-string templating (default)
rendered = prompt.with_engine("f-string").render(**kwargs)

# Mustache templating
rendered = prompt.with_engine("mustache").render(**kwargs)

# Jinja2 templating
rendered = prompt.with_engine("jinja2").render(**kwargs)
```

## Working with Labels

Labels provide stable references to specific revisions:

```python
from patronus import context

client = context.get_api_client().prompts

# Add audience-specific labels
client.add_label(
    prompt_id="prompt_123",
    revision=3,
    label="technical-audience"
)

# Update label to point to a new revision
client.add_label(
    prompt_id="prompt_123",
    revision=5,
    label="technical-audience"
)

# Add environment label
client.add_label(
    prompt_id="prompt_456",
    revision=2,
    label="production"
)
```

## Metadata Usage

Prompt revisions support arbitrary metadata:

```python
from patronus.prompts import Prompt, push_prompt, load_prompt

# Create with metadata
prompt_with_meta = Prompt(
    name="research/data-analysis/summarize-findings",
    body="Analyze the {data_type} data and summarize the key {metric_type} trends in {time_period}.",
    metadata={
        "models": ["gpt-4", "claude-3"],
        "created_by": "data-team",
        "tags": ["data", "analysis"]
    }
)

loaded_prompt = push_prompt(prompt_with_meta)

# Access metadata
prompt = load_prompt(name="research/data-analysis/summarize-findings")
supported_models = prompt.metadata.get("models", [])
creator = prompt.metadata.get("created_by", "unknown")

print(f"Prompt supports models: {', '.join(supported_models)}")
print(f"Created by: {creator}")
```

## Using Multiple Prompts Together

Complex applications often use multiple prompts together:

```python
import patronus
from patronus.prompts import load_prompt
import openai

patronus.init()

# Load different prompt components
system_prompt = load_prompt(name="support/chat/system")
user_query_template = load_prompt(name="support/chat/user-message")
response_formatter = load_prompt(name="support/chat/response-format")

# Create OpenAI client
client = openai.OpenAI()

# Combine the prompts in a chat completion
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": system_prompt.render(
            product_name="CloudWorks Pro",
            available_features=["file sharing", "collaboration", "automation"],
            knowledge_cutoff="2024-05-01"
        )},
        {"role": "user", "content": user_query_template.render(
            user_name="Alex",
            user_tier="premium",
            user_query="How do I share files with external users?"
        )}
    ],
    temperature=0.7,
    max_tokens=500
)

# Post-process the response using another prompt
formatted_response = response_formatter.render(
    raw_response=response.choices[0].message.content,
    user_name="Alex",
    add_examples=True
)
```

## Naming Conventions

Use a descriptive, hierarchical naming structure similar to file paths. This makes prompts easier to organize, find, and manage:

```
[domain]/[use-case]/[component]/[prompt-type]
```

Where `[prompt-type]` indicates the intended role of the prompt in an LLM conversation (optional but recommended):

- `system` - Sets the overall behavior, persona, or context for the model
- `instruction` - Provides specific instructions for a task
- `user` - Represents a user message template
- `assistant` - Template for assistant responses
- `few-shot` - Contains examples of input/output pairs

Examples:

- `support/troubleshooting/diagnostic-questions/system`
- `marketing/email-campaigns/follow-up-template/instruction`
- `dev/code-generation/python-function/instruction`
- `finance/report/quarterly-analysis`
- `content/blog-post/technical-tutorial/few-shot`
- `legal/contracts/terms-of-service-v2/system`

Including the prompt type in the name helps team members quickly understand the intended usage context in multi-prompt conversations.

### Consistent Prefixes

Use consistent prefixes for prompts that work together in the same feature:

```
# Onboarding chat prompts share the prefix onboarding/chat/
onboarding/chat/welcome/system
onboarding/chat/questions/user
onboarding/chat/intro/assistant

# Support classifier prompts
support/classifier/system
support/classifier/categories/instruction
```

This approach simplifies filtering and management of related prompts, making it easier to maintain and evolve complete prompt flows as your library grows.

## Configuration

The default template engine can be configured during initialization:

```python
import patronus

patronus.init(
    # Default template engine for all prompts
    prompt_templating_engine="mustache"
)
```

For additional configuration options, see the [Configuration](../configuration.md) page.

## Using with LLMs

Prompts can be used with any LLM provider:

```python
import patronus
from patronus.prompts import load_prompt
import anthropic

patronus.init()

system_prompt = load_prompt(name="support/knowledge-base/technical-assistance")

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-3-opus-20240229",
    system=system_prompt.render(
        product_name="CloudWorks Pro",
        user_tier="enterprise",
        available_features=["advanced monitoring", "auto-scaling", "SSO integration"]
    ),
    messages=[
        {"role": "user", "content": "How do I configure the load balancer for high availability?"}
    ]
)
```

## Additional Resources

While the SDK provides high-level, convenient access to Patronus functionality, you can also use the lower-level APIs for more direct control:

- [REST API documentation](https://docs.patronus.ai/docs/api_ref) - For direct HTTP access to the Patronus platform
- [Patronus API Python library](https://github.com/patronus-ai/patronus-api-python) - A typed Python client for the REST API with both synchronous and asynchronous support
