# Installation

The Patronus SDK provides tools for evaluating, monitoring, and improving LLM applications.

## Requirements

- Python 3.9 or higher
- A package manager (uv or pip)

## Basic Installation

### Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver:

```bash
uv add patronus
```

### Using pip

```bash
pip install patronus
```

## Optional Dependencies

### For Experiments

To use Patronus experiments functionality (including pandas support):

```bash
# Using uv
uv add "patronus[experiments]"

# Using pip
pip install "patronus[experiments]"
```

## Quick Start with Examples

If you'd like to see Patronus in action quickly, check out our [examples](../examples/index.md). These examples demonstrate how to use Patronus with various LLM frameworks and APIs.

For instance, to run the Smolagents weather example:

```bash
# Export required API keys
export PATRONUS_API_KEY=your-api-key
export OPENAI_API_KEY=your-api-key

# Run the example with uv
uv run --no-cache --with "patronus-examples[smolagents]" \
    -m patronus_examples.tracking.smolagents_weather
```

See the [examples documentation](../examples/index.md) for more detailed information on running and understanding the available examples.
