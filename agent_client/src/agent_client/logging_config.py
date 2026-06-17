"""
agent_client/logging_config.py
═══════════════════════════════
Dual-stream logging setup.

Writes TWO streams to agent_system.log with clear prefixes:
  [CLIENT] — orchestration logs from the agent process
  [SERVER] — forwarded MCP notification logs from the server

Also mirrors everything to stdout so operators can tail the console.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Constants
LOG_FILE = Path("mcp_agent_system.log")
TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S"

# Custom formatters


class ClientFormatter(logging.Formatter):
    """Formats client-side log records with [CLIENT] prefix."""

    def format(self, record: logging.LogRecord) -> str:
        ts = self.formatTime(record, TIMESTAMP_FMT)
        return f"[{ts}] [CLIENT] [{record.levelname}] {record.getMessage()}"


class ServerFormatter(logging.Formatter):
    """Formats server-forwarded log records with [SERVER] prefix."""

    def format(self, record: logging.LogRecord) -> str:
        ts = self.formatTime(record, TIMESTAMP_FMT)
        return f"[{ts}] [SERVER] [{record.levelname}] {record.getMessage()}"


# Logger factory


def setup_client_logger(name: str = "agent_client") -> logging.Logger:
    """
    Return a logger that writes [CLIENT] entries to stdout and
    mcp_agent_system.log.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger  # already configured

    formatter = ClientFormatter()

    # stdout handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)

    # file handler (append mode so multi-turn runs accumulate)
    file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.propagate = False

    return logger


def get_server_log_writer() -> logging.Logger:
    """
    Return a logger that writes [SERVER] entries to stdout and
    mcp_agent_system.log.
    Called by the MCP log_handler callback when server notification
    logs arrive.
    """
    logger = logging.getLogger("server_forwarded")
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    formatter = ServerFormatter()

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.propagate = False

    return logger
