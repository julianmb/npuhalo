# Contributing

Contributions that improve reproducibility, safety, documentation, or measured
NPU utility are welcome.

## Development Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[dev]"
make lint
make test
```

Python 3.12 is the supported development version.

## Pull Requests

- Keep changes focused and explain the measured or user-visible effect.
- Add tests for behavior changes where practical.
- Do not commit model weights, credentials, local configuration, raw private prompts, or machine-specific paths.
- Bind development services to `127.0.0.1` unless remote exposure is the explicit subject of the change.
- Treat generated code and shell commands as untrusted; use a disposable container or VM for adversarial workloads.

## Benchmark Contributions

Include the model identifier and hash, runtime and commit, driver and firmware
versions, prompt or dataset version, seeds, commands, and raw aggregate output.
Clearly distinguish measured values from projections or assumptions. Sanitize
hostnames, usernames, temporary paths, and unrelated environment details before
submitting telemetry.
