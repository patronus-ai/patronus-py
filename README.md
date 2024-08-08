# Patronus Python SDK

Patronus Python SKD is in alpha version right now.

## Installation

Right now you need to clone repository and install package directly.

```shell
poetry install
```

## How to use

Please see [examples](examples) directory.

All samples require Patronus AI API key. Some of the also require Open AI API Key.
Examples that use LevenshteinScorer also need external dependency.

```shell
export PATRONUSAI_API_KEY=<PROVIDE KEY>
export OPENAI_API_KEY=<PROVIDE KEY>

pip install Levenshtein

python examples/ex_0_hello_world.py
```
