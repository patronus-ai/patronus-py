# Troubleshooting

Common issues and solutions when using Patronus observability features.

## Connection Issues

### StatusCode.DeadlineExceeded Error

**Problem:**

You see `StatusCode.DEADLINE_EXCEEDED` errors when using the SDK with tracing enabled. Traces and logs are not appearing in the Patronus platform.

**Solution:**

Increase the OTLP exporter timeout by setting the environment variable:

```bash
export OTEL_EXPORTER_OTLP_TIMEOUT=30
```

Then restart your application. This sets the timeout to 30 seconds, giving the client more time to establish a connection and send data.

**Additional Steps:**

- Verify your network connection is stable
- Check that you can reach `https://otel.patronus.ai:4317`
- Ensure no firewall or proxy is blocking the connection
