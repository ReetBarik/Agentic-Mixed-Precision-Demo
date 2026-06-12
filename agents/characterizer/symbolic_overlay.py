"""LLM-driven symbolic idiom detection — optional, never gates the pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import anthropic

from agents.characterizer.profile import SymbolicHint
from agents.config import PipelineConfig

_TIMEOUT_SECONDS = 10


def analyze(
    source_files: list[str],
    kernel_name: str,
    cfg: PipelineConfig,
) -> list[SymbolicHint]:
    """Detect numerically unstable idioms in the kernel source.

    Returns an empty list on any error — never raises.
    """
    kernel_source = _collect_source(source_files)
    if not kernel_source.strip():
        return []

    prompt_template = (Path(__file__).parent / "prompts" / "symbolic_overlay.txt").read_text(
        encoding="utf-8"
    )
    prompt = prompt_template.format(kernel_source=kernel_source)

    client = anthropic.Anthropic(
        base_url=cfg.base_url,
        api_key=cfg.auth_token,
    )

    response = client.messages.create(
        model=cfg.model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
        timeout=_TIMEOUT_SECONDS,
    )

    raw_text = ""
    for block in response.content:
        if hasattr(block, "text"):
            raw_text += block.text

    return _parse_hints(raw_text.strip())


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _collect_source(source_files: list[str]) -> str:
    parts = []
    for path in source_files:
        try:
            parts.append(Path(path).read_text(encoding="utf-8"))
        except OSError:
            pass
    return "\n".join(parts)


def _parse_hints(text: str) -> list[SymbolicHint]:
    start = text.find("[")
    if start == -1:
        return []
    raw_list = json.loads(text[start:])
    hints = []
    for item in raw_list:
        hints.append(
            SymbolicHint(
                idiom=item.get("idiom", "unknown"),
                location=item.get("location", ""),
                severity=item.get("severity", "medium"),
                suggested_rewrite=item.get("suggested_rewrite", ""),
            )
        )
    return hints
