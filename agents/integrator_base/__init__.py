"""Shared machinery for interop-shim integrators (tracked / dd / ff).

Factored out of ``agents/tracked_integrator/agent.py`` so every integrator that
turns a target library into something callable with an instrumented / extended
numeric type can reuse the same plumbing, while keeping its own *rules* and
*system prompt* target-specific:

* :mod:`agents.integrator_base.cache` — ``SOURCE_HASH`` staleness cache
  (header-dir hash ⊕ ruleset hash), and the shim-write/cache-hit flow.
* :mod:`agents.integrator_base.llm` — Anthropic streaming shim, code-fence
  stripping, target-header embedding helpers, and a bounded generate/accept
  retry loop with cache-bypass.
* :mod:`agents.integrator_base.c8` — compiler-error-driven type-boundary
  patching (``derive_c8_patch``) parameterized on the target scalar type name,
  plus the generic difflib unified-diff synthesizer.

The rules (classification ruleset / system prompt) stay in each integrator
package (e.g. ``agents/tracked_integrator/system_prompt.txt``) and are passed
into these helpers, so the ``SOURCE_HASH`` correctly invalidates when *either*
the target headers or that integrator's ruleset changes.
"""
