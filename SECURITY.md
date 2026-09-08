# Security Policy

## Supported Version

Security fixes are applied to the latest revision of the `main` branch while the
project remains pre-1.0 research software.

## Threat Model

`npuhalo` launches local inference services and contains evaluation scripts that
can execute generated Python and shell commands. A temporary directory and a
timeout are not security isolation. Run untrusted workloads only inside a
disposable container or virtual machine with restricted filesystem and network
access.

Services bind to localhost by default. If remote access is required, place them
behind an authenticated TLS reverse proxy and configure request, concurrency,
and timeout limits. Never expose the experimental services directly to an
untrusted network.

## Reporting a Vulnerability

Use GitHub's private vulnerability reporting feature for this repository. Do
not open a public issue containing exploit details, secrets, or sensitive logs.
Include affected files, reproduction steps, impact, and a suggested mitigation
when available.
