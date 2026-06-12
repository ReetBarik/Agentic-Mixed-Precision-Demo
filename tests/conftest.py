"""Shared pytest configuration and fixtures."""

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "llm: requires a live LLM (Argo proxy)")
    config.addinivalue_line("markers", "kokkos: requires a Kokkos installation")
