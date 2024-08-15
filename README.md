# Patronus LLM Evaluation library


Patronus is a Python library developed by [Patronus AI](https://www.patronus.ai/)
that provides a robust framework and utility functions for evaluating Large Language Models (LLMs).
This library simplifies the process of running and scoring evaluations across different LLMs,
making it easier for developers to benchmark model performance on various tasks.

**Note:** This library is currently in **beta** and is not stable. The APIs may change in future releases.

**Note:** This library requires Python 3.11 or greater.

## Features

- **Modular Evaluation Framework:** Easily plug in different models and evaluation/scoring mechanisms.
- **Seamless Integration with Patronus AI Platform:** Effortlessly connect with the Patronus AI platform to run evaluations and export results.
- **Custom Evaluators:** Use built-in evaluators, create your own based on various scoring methods, or leverage our state-of-the-art remote evaluators.

## Documentation

[//]: # (TODO Update documentation link once it's up and live)
For detailed documentation, including API references and advanced usage, please visit our [documentation](https://docs.patronus.ai/).

## Installation

To get started with Patronus, clone the repository and install the package using Poetry:

```shell
git clone https://github.com/patronus-ai/patronus-py
cd patronus-py
poetry install
```

## Usage

### Prerequisites

Before running any examples, make sure you have the following API keys:

- **Patronus AI API Key:** Required for all examples.
- **OpenAI API Key:** Required for some examples that utilize OpenAI's services.

You can set these keys as environment variables:

```shell
export PATRONUSAI_API_KEY=<YOUR_PATRONUSAI_API_KEY>
export OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>
```

### Running Examples

Patronus comes with several example scripts to help you understand how to use the library. These examples can be found in the [examples](examples) directory.

**Note:** Some examples require additional dependencies. For instance:
- If you are using an evaluator that depends on the `Levenshtein` scoring method, you need to install the `Levenshtein` package:

  ```shell
  pip install Levenshtein
  ```

- If you are using examples that integrate with OpenAI, you need to install the `openai` package:

  ```shell
  pip install openai
  ```

You can then run an example script like this:

```shell
python examples/ex_0_hello_world.py
```
