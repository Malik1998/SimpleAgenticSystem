import logging

import pytest

from simple_agentic_system.observability import LoggingTracer, NullTracer


def test_null_tracer_is_noop():
    tracer = NullTracer()
    with tracer.span("x") as span:
        span.set_output("ignored")
        span.set_error("ignored")


def test_logging_tracer_logs_span_start_and_output(caplog):
    tracer = LoggingTracer()
    with caplog.at_level(logging.INFO, logger="simple_agentic_system.trace"):
        with tracer.span("my-span", kind="tool", foo="bar") as span:
            span.set_output(42)
    assert "my-span" in caplog.text
    assert "42" in caplog.text


def test_logging_tracer_logs_error(caplog):
    tracer = LoggingTracer()
    with caplog.at_level(logging.WARNING, logger="simple_agentic_system.trace"):
        with tracer.span("failing-span") as span:
            span.set_error("boom")
    assert "boom" in caplog.text


def test_logging_tracer_logs_and_reraises_exceptions(caplog):
    tracer = LoggingTracer()
    with caplog.at_level(logging.ERROR, logger="simple_agentic_system.trace"), pytest.raises(ValueError):
        with tracer.span("raising-span"):
            raise ValueError("kaboom")
    assert "failed" in caplog.text
