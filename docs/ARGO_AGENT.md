# Direct Argo LLM Client (No MCP)

This repo now includes a minimal direct Argo chat client in `llm_agent/`.

## Prerequisites

- A running local Argo proxy endpoint on JLSE (example: `http://127.0.0.1:8092`).
- `ARGO_USERNAME` exported in the shell.
- Python 3.12 for the agent workflow (`python3` may be 3.6 on JLSE).
- Python dependency:

```bash
python3.12 -m pip install --user -r requirements-argo-agent.txt
```

## Start proxy

Use the helper script from this repo:

```bash
bash scripts/setup_argo_proxy.sh --start-port 8092
```

Keep that terminal open.

## Install dependencies correctly on JLSE

Do not route `pip` through the Argo proxy.

```bash
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
python3.12 -m pip install --user -r requirements-argo-agent.txt
```

Then set proxy vars for Argo requests:

```bash
export HTTP_PROXY=http://127.0.0.1:8092
export HTTPS_PROXY=http://127.0.0.1:8092
export ARGO_USERNAME=<your_anl_username>
```

## Smoke test the Python client

From another terminal:

```bash
export ARGO_USERNAME=<your_anl_username>
python3.12 -m llm_agent.cli --base-url http://127.0.0.1:8092 --prompt "Reply with OK"
```

## Generate a patch proposal (LLM only)

This step asks Argo to propose a unified diff for a target driver using
`targets.json` + the implementation source file. It does not apply the patch.

```bash
python3.12 -m llm_agent.patch_proposer \
  --driver ddilog \
  --base-url http://127.0.0.1:8092 \
  --min-digits 10
```

The command prints the generated patch file path, typically under:
`experiments/<driver>/generated/proposed_patch_<timestamp>.patch`

## Verify a proposed patch (isolated worktree)

Apply and evaluate a generated patch in a temporary git worktree:

```bash
python3.12 -m llm_agent.patch_verify \
  --patch-file experiments/ddilog/generated/proposed_patch_<timestamp>.patch \
  --driver ddilog \
  --batch 10 \
  --seed 123 \
  --min-digits 10
```

The command prints a JSON report path, typically:
`experiments/<driver>/generated/verify_report_<timestamp>.json`

Exit codes:

- `0`: patch applied and precision check passed
- `1`: patch applied, but precision check failed
- `2`: patch apply/setup/runtime error

## One-shot episode (propose -> verify)

Run a complete single attempt:

```bash
python3.12 -m llm_agent.run_episode \
  --driver ddilog \
  --base-url http://127.0.0.1:8092 \
  --min-digits 10 \
  --batch 10 \
  --seed 123
```

This prints a final JSON verdict with:

- propose exit code
- verify exit code
- pass/fail
- patch file path
- verify report path

Defaults:

- Primary model: `claudeopus46`
- Fallback model: `gpt4turbo` (used if the primary model is rejected)

You can override models:

```bash
python3.12 -m llm_agent.cli --base-url http://127.0.0.1:8092 --model gpt4turbo
```

## Environment knobs

- `ARGO_USERNAME` (required)
- `ARGO_PROXY_BASE_URL` (optional alternative to `--base-url`)
- `ARGO_PROXY_PORT` (optional, default `8092` when base URL is not provided)

## Known-good request notes

- `gpt4turbo` is validated on this setup.
- Some display-name models returned by `/v1/models` may not be accepted directly by `/v1/chat/completions`.
- Some model adapters reject `max_tokens`; omit it unless needed.

## Troubleshooting

- `bind: address already in use` from proxy startup:
  - retry with a different port: `bash scripts/setup_argo_proxy.sh --start-port 8092`
- `ModuleNotFoundError: requests` when running CLI:
  - install and run with the same interpreter (`python3.12`)
- `pip` fails with proxy/tunnel errors:
  - unset `HTTP_PROXY`/`HTTPS_PROXY` before install
- Argo `400 invalid model`:
  - switch to a known-good id, e.g. `--model gpt4turbo`

