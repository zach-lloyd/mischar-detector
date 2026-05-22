"""Structured logging configuration.

Configures structlog with JSON output for machine readability and a
human-friendly console renderer for development.
"""

from __future__ import annotations

import logging

import structlog


def configure_logging(level: str = "INFO") -> None:
    """Set up structlog with the given log level to produce structured key-value
    log output instead of plain strings.

    Call once at application startup (CLI entrypoint or evaluation runner).
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    structlog.configure(
        processors=[
            # Lets you bind context that automatically attaches to all subsequent log 
            # messages
            structlog.contextvars.merge_contextvars,
            # Adds [info], [warning], [error] to each message
            structlog.processors.add_log_level,
            # Includes stack trace info when you log an exception
            structlog.processors.StackInfoRenderer(),
            # Automatically captures exception details on error-level logs
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            # Formats everything in a human-readable way for your terminal. 
            # In production, swap for JSONRenderer
            structlog.dev.ConsoleRenderer(),
        ],
        # Set log_level to DEBUG in config.py if you want verbose output to assist
        # with debugging
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a named logger instance. Each module calls log = get_logger("stage_name") 
    so you can tell which part of the pipeline produced each log line."""
    return structlog.get_logger(name)
