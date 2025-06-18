# Observability Configuration

## Exporter Protocols

The SDK supports two OTLP exporter protocols:

| Protocol | Value           | Default Endpoint                | Available Ports |
|----------|-----------------|---------------------------------|-----------------|
| gRPC     | `grpc`          | `https://otel.patronus.ai:4317` | 4317            |
| HTTP     | `http/protobuf` | `https://otel.patronus.ai:4318` | 4318, 443       |

## Configuration Methods

### 1. Patronus Configuration

```python
patronus.init(
    otel_endpoint="https://otel.patronus.ai:4318",
    otel_exporter_otlp_protocol="http/protobuf"
)
```

```yaml
# patronus.yaml
otel_endpoint: "https://otel.patronus.ai:4318"
otel_exporter_otlp_protocol: "http/protobuf"
```

```bash
export PATRONUS_OTEL_ENDPOINT="https://otel.patronus.ai:4318"
export PATRONUS_OTEL_EXPORTER_OTLP_PROTOCOL="http/protobuf"
```

### 2. OpenTelemetry Environment Variables

```bash
# General (applies to all signals)
export OTEL_EXPORTER_OTLP_PROTOCOL="grpc"

# Signal-specific
export OTEL_EXPORTER_OTLP_TRACES_PROTOCOL="http/protobuf"
export OTEL_EXPORTER_OTLP_LOGS_PROTOCOL="grpc"
```

## Configuration Priority

1. Function parameters
2. Environment variables (`PATRONUS_OTEL_EXPORTER_OTLP_PROTOCOL`)
3. Configuration file (`patronus.yaml`)
4. `OTEL_EXPORTER_OTLP_TRACES_PROTOCOL` / `OTEL_EXPORTER_OTLP_LOGS_PROTOCOL`
5. `OTEL_EXPORTER_OTLP_PROTOCOL`
6. Default: `grpc`

## Endpoint Configuration

### Custom Endpoints

```python
patronus.init(
    otel_endpoint="https://collector.example.com:4317",
    otel_exporter_otlp_protocol="grpc"
)
```

### Connection Security

Security is determined by the URL scheme for both gRPC and HTTP protocols:

- `https://` - Secure connection (TLS)
- `http://` - Insecure connection

```python
# Secure gRPC
patronus.init(otel_endpoint="https://collector.example.com:4317")

# Insecure gRPC
patronus.init(otel_endpoint="http://collector.example.com:4317")

# Secure HTTP
patronus.init(
    otel_endpoint="https://collector.example.com:4318",
    otel_exporter_otlp_protocol="http/protobuf"
)

# Insecure HTTP
patronus.init(
    otel_endpoint="http://collector.example.com:4318",
    otel_exporter_otlp_protocol="http/protobuf"
)
```

### HTTP Path Handling

For HTTP protocol, paths are automatically appended:

- Traces: `<endpoint>/v1/traces`
- Logs: `<endpoint>/v1/logs`

## Examples

### HTTP Protocol with Custom Endpoint

```python
patronus.init(
    otel_endpoint="http://internal-collector:8080",
    otel_exporter_otlp_protocol="http/protobuf"
)
```

### HTTP Protocol on Standard HTTPS Port

```python
patronus.init(
    otel_endpoint="https://otel.example.com:443",
    otel_exporter_otlp_protocol="http/protobuf"
)
```

### gRPC with Insecure Connection

```python
patronus.init(
    otel_endpoint="http://internal-collector:4317",
    otel_exporter_otlp_protocol="grpc"
)
```

### Mixed Protocols

```bash
export OTEL_EXPORTER_OTLP_TRACES_PROTOCOL="http/protobuf"
export OTEL_EXPORTER_OTLP_LOGS_PROTOCOL="grpc"
```
