"""
This module provides exporter selection functionality for OpenTelemetry traces and logs.
It handles protocol resolution based on Patronus configuration and standard OTEL environment variables.
"""

import os
from typing import Optional, Literal

from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter as OTLPLogExporterGRPC
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter as OTLPSpanExporterGRPC
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter as OTLPLogExporterHTTP
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as OTLPSpanExporterHTTP
from opentelemetry.sdk._logs.export import LogExporter
from opentelemetry.sdk.trace.export import SpanExporter


def _resolve_otlp_protocol(protocol_override: Optional[str]) -> Literal["grpc", "http/protobuf"]:
    """
    Resolve the OTLP protocol to use based on configuration and environment variables.

    Priority order:
    1. protocol_override parameter (from Patronus config)
    2. OTEL_EXPORTER_OTLP_TRACES_PROTOCOL or OTEL_EXPORTER_OTLP_LOGS_PROTOCOL (signal-specific)
    3. OTEL_EXPORTER_OTLP_PROTOCOL (general)
    4. Default: "grpc"

    Args:
        protocol_override: Protocol specified in Patronus configuration

    Returns:
        Protocol string: "grpc" or "http/protobuf"
    """
    if protocol_override in ("grpc", "http/protobuf"):
        return protocol_override

    # Check standard OTEL environment variables
    # Note: We check both traces and logs protocols as they might differ
    traces_protocol = os.environ.get("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL")
    logs_protocol = os.environ.get("OTEL_EXPORTER_OTLP_LOGS_PROTOCOL")
    general_protocol = os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL")

    # Use traces protocol if available, otherwise fall back to general
    resolved_protocol = traces_protocol or logs_protocol or general_protocol

    if resolved_protocol:
        resolved_protocol = resolved_protocol.strip().lower()
        if resolved_protocol in ("grpc", "http/protobuf"):
            return resolved_protocol

    # Default to gRPC
    return "grpc"


def create_trace_exporter(endpoint: str, api_key: str, protocol: Optional[str] = None) -> SpanExporter:
    """
    Create a configured trace exporter instance.

    Args:
        endpoint: The OTLP endpoint URL
        api_key: Authentication key for Patronus services
        protocol: OTLP protocol override from Patronus configuration

    Returns:
        Configured trace exporter instance
    """
    resolved_protocol = _resolve_otlp_protocol(protocol)

    if resolved_protocol == "http/protobuf":
        # For HTTP exporter, ensure endpoint has the correct path
        if not endpoint.endswith("/v1/traces"):
            endpoint = endpoint.rstrip("/") + "/v1/traces"
        return OTLPSpanExporterHTTP(endpoint=endpoint, headers={"x-api-key": api_key})
    else:
        # For gRPC exporter, determine if connection should be insecure based on URL scheme
        is_insecure = endpoint.startswith("http://")
        return OTLPSpanExporterGRPC(endpoint=endpoint, headers={"x-api-key": api_key}, insecure=is_insecure)


def create_log_exporter(endpoint: str, api_key: str, protocol: Optional[str] = None) -> LogExporter:
    """
    Create a configured log exporter instance.

    Args:
        endpoint: The OTLP endpoint URL
        api_key: Authentication key for Patronus services
        protocol: OTLP protocol override from Patronus configuration

    Returns:
        Configured log exporter instance
    """
    resolved_protocol = _resolve_otlp_protocol(protocol)

    if resolved_protocol == "http/protobuf":
        # For HTTP exporter, ensure endpoint has the correct path
        if not endpoint.endswith("/v1/logs"):
            endpoint = endpoint.rstrip("/") + "/v1/logs"
        return OTLPLogExporterHTTP(endpoint=endpoint, headers={"x-api-key": api_key})
    else:
        # For gRPC exporter, determine if connection should be insecure based on URL scheme
        is_insecure = endpoint.startswith("http://")
        return OTLPLogExporterGRPC(endpoint=endpoint, headers={"x-api-key": api_key}, insecure=is_insecure)
