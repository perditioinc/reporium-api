"""OpenTelemetry setup for Cloud Trace export."""

import logging
import os

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logger = logging.getLogger(__name__)


def init_telemetry() -> TracerProvider | None:
    """Initialize OTel with Cloud Trace exporter if enabled.

    Feature-flagged via ``OTEL_ENABLED=1``.  In dev (default ``0``) this is a
    complete no-op so there is zero overhead when tracing is not wanted.
    """
    if os.getenv("OTEL_ENABLED", "0") != "1":
        return None  # No-op in dev

    from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor  # noqa: F401 — used by caller

    provider = TracerProvider()
    processor = BatchSpanProcessor(CloudTraceSpanExporter())
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

    logger.info("OpenTelemetry initialised with Cloud Trace exporter")
    return provider
