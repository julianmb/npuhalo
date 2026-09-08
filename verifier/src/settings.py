"""Shared runtime settings for the live-verifier stack.

Reads key=value pairs from the config file (default: config.example.env
next to the verifier root, or the path in $NPUHALO_CONFIG) and
lets real environment variables override file values.
"""

import os
from typing import Dict

_KNOWN_KEYS = (
    "ORNITH_ENDPOINT",
    "LFM_NPU_ENDPOINT",
    "LFM_MODEL_ID",
    "VERIFIER_BACKEND",
    "CHECKPOINT_TOKEN_LIMIT",
    "VERIFIER_K",
    "VERIFIER_TEMPERATURE",
)

_DEFAULTS: Dict[str, str] = {
    "ORNITH_ENDPOINT": "http://127.0.0.1:8012/v1",
    "LFM_NPU_ENDPOINT": "http://127.0.0.1:13305/v1",
    "LFM_MODEL_ID": "LiquidAI/LFM2.5-1.2B-Thinking",
    "VERIFIER_BACKEND": "auto",
    "CHECKPOINT_TOKEN_LIMIT": "250",
    "VERIFIER_K": "3",
    "VERIFIER_TEMPERATURE": "0.7",
}


def load_settings(config_path: str = None) -> Dict[str, str]:
    settings = dict(_DEFAULTS)

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if config_path is None:
        candidates = [os.environ.get("NPUHALO_CONFIG", ""),
                      os.path.join(base_dir, "config.env"),
                      os.path.join(base_dir, "config.example.env")]
        config_path = next(
            (os.path.abspath(p) for p in candidates if p and os.path.exists(p)),
            None,
        )

    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                settings[key.strip()] = value.strip()

    for key in _KNOWN_KEYS:
        if os.environ.get(key):
            settings[key] = os.environ[key]

    return settings
