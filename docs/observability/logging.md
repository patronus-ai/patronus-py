# Logging

Logging is an essential feature of the Patronus SDK that allows you to record events, debug information, and track the execution of your LLM applications.
This page covers how to set up and use logging in your code.

!!! tip "Configuration"

    For information about configuring observability features, including exporter protocols and endpoints, see the [Observability Configuration](configuration.md) guide.

## Getting Started with Logging

The Patronus SDK provides a simple logging interface that integrates with Python's standard logging module while also automatically exporting logs to the Patronus AI Platform:

```python
import patronus

patronus.init()
log = patronus.get_logger()

# Basic logging
log.info("Processing user query")

# Different log levels are available
log.debug("Detailed debug information")
log.warning("Something might be wrong")
log.error("An error occurred")
log.critical("System cannot continue")
```

## Configuring Console Output

By default, Patronus logs are sent to the Patronus AI Platform but are not printed to the console.
To display logs in your console output, you can add a standard Python logging handler:

```python
import sys
import logging
import patronus

patronus.init()
log = patronus.get_logger()

# Add a console handler to see logs in your terminal
console_handler = logging.StreamHandler(sys.stdout)
log.addHandler(console_handler)

# Now logs will appear in both console and Patronus Platform
log.info("This message appears in the console and is sent to Patronus")
```

You can also customize the format of console logs:

```python
import sys
import logging
import patronus

patronus.init()
log = patronus.get_logger()

formatter = logging.Formatter('[%(asctime)s] %(levelname)-8s: %(message)s')

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
log.addHandler(console_handler)

# Logs will now include timestamp and level
log.info("Formatted log message")
```

## Advanced Configuration

Patronus integrates with Python's logging module, allowing for advanced configuration options. The SDK uses two main loggers:

- `patronus.sdk` - For client-emitted messages that are automatically exported to the Patronus AI Platform
- `patronus.core` - For library-emitted messages related to the SDK's internal operations

Here's how to configure these loggers using standard library methods:

```python
import logging
import patronus

# Initialize Patronus before configuring logging
patronus.init()

# Configure the root Patronus logger
patronus_root_logger = logging.getLogger("patronus")
patronus_root_logger.setLevel(logging.WARNING)  # Set base level for all Patronus loggers

# Add a console handler with custom formatting
console_handler = logging.StreamHandler()
formatter = logging.Formatter(
    fmt='[%(asctime)s] %(levelname)-8s %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console_handler.setFormatter(formatter)
patronus_root_logger.addHandler(console_handler)

# Configure specific loggers
patronus_core_logger = logging.getLogger("patronus.core")
patronus_core_logger.setLevel(logging.WARNING)  # Only show warnings and above for internal SDK messages

patronus_sdk_logger = logging.getLogger("patronus.sdk")
patronus_sdk_logger.setLevel(logging.INFO)  # Show info and above for your application logs
```

## Logging with Traces

Patronus logging integrates seamlessly with the tracing system, allowing you to correlate logs with specific spans in your application flow:

```python
import patronus
from patronus import traced, start_span

patronus.init()
log = patronus.get_logger()

@traced()
def process_user_query(query):
    log.info("Processing query")

    with start_span("Query Analysis"):
        log.info("Analyzing query intent")
        ...

    with start_span("Response Generation"):
        log.info("Generating LLM response")
        ...

    return "Response to: " + query

# Logs will be associated with the appropriate spans
result = process_user_query("Tell me about machine learning")
```
