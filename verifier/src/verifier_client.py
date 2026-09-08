#!/usr/bin/env python3
"""
NPU / Local Verifier Client for LFM2.5-1.2B-Thinking.

Two backends:
  * "npu"   : NPU endpoint (Lemonade / FastFlowLM fronting the LFM model) via
              OpenAI-compatible /chat/completions; batched K sampling & majority vote.
  * "local" : in-process PyTorch reference implementation.

VERIFIER_BACKEND in config.env / env controls selection:
  * "auto"  (default) -- use the NPU endpoint when reachable; else fall back to local.
  * "npu"   -- NPU endpoint only; raises if unreachable.
  * "local" -- in-process PyTorch only.

Implements batched K-sample sampling with majority voting for:
  CONTINUE / SUSPECT / ABORT
and parses quoted evidence before the verdict tag.
"""

import re
import time
import urllib.error
import urllib.request
from collections import Counter
from typing import Dict, Any, Tuple, Optional, List

DEFAULT_MODEL_ID = "LiquidAI/LFM2.5-1.2B-Thinking"

try:
    from .settings import load_settings
except ImportError:  # Support direct imports used by source-tree benchmark scripts.
    from settings import load_settings

_SETTINGS = load_settings()


class NPUVerifierClient:
    """OpenAI-client verifier talking to an NPU endpoint (no local weights)."""

    MAX_MODELS = 20

    def __init__(self, endpoint: str):
        self.endpoint = endpoint.rstrip("/")
        self.backend = "npu"
        self._chat_url = self._endpoint_to_chat_completions(self.endpoint)
        self._models_url = self._endpoint_to_models(self.endpoint)

    @staticmethod
    def _endpoint_to_chat_completions(endpoint: str) -> str:
        if endpoint.endswith("/v1/chat/completions"):
            return endpoint
        if endpoint.endswith("/chat/completions"):
            return endpoint
        return endpoint + "/chat/completions"

    @staticmethod
    def _endpoint_to_models(endpoint: str) -> str:
        if endpoint.endswith("/v1/chat/completions"):
            return endpoint[: -len("/chat/completions")] + "/models"
        if endpoint.endswith("/chat/completions"):
            return endpoint[: -len("/chat/completions")] + "/models"
        if endpoint.endswith("/models") or endpoint.endswith("/model"):
            return endpoint
        if endpoint.endswith("/v1/models"):
            return endpoint
        return endpoint + "/models"

    @classmethod
    def healthy(cls, endpoint: str) -> bool:
        try:
            tmp = cls(endpoint)
            _ = tmp.available_models(timeout=2.0)
            return True
        except Exception:
            return False

    def available_models(self, timeout: float = 5.0) -> List[str]:
        req = urllib.request.Request(self._models_url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            import json
            body = json.loads(resp.read().decode("utf-8"))
        data = body.get("data", []) if isinstance(body, dict) else body
        ids = []
        for d in (data or []):
            mid = d.get("id") if isinstance(d, dict) else None
            if mid:
                ids.append(mid)
        return ids[: self.MAX_MODELS]

    PREFERRED_MODEL_ID = "LiquidAI/LFM2.5-1.2B-Thinking"

    def _pick_model_id(self) -> Optional[str]:
        ids = self.available_models()
        for probe in ("LFM2.5-TK", "LFM2.5-THINKING", "LFM2.5", "LFM2-TK", "LFM2", "LFM", "LIQUIDAI"):
            for mid in ids:
                if probe.upper() in mid.upper():
                    return mid
        return ids[0] if ids else None

    def evaluate_checkpoint(
        self,
        task_prompt: str,
        trajectory: str,
        prompt_template: str,
        k_samples: int = 3,
        temperature: float = 0.7,
        timeout: float = 90.0,
    ) -> Dict[str, Any]:
        model_id = self._pick_model_id()
        if model_id is None:
            raise RuntimeError(f"verifier NPU endpoint returned no models: {self.endpoint}")

        messages = [
            {"role": "user", "content": prompt_template.format(
                task=task_prompt, trajectory=trajectory)}
        ]

        payload = {
            "model": model_id,
            "messages": messages,
            "n": k_samples,
            "max_tokens": 25,
            "temperature": float(temperature) if temperature > 0.0 else 0.0,
            "stream": False,
        }
        import json
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._chat_url, data=data,
            headers={"Content-Type": "application/json"},
        )
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        latency_ms = (time.time() - t0) * 1000

        choices = body.get("choices") or []
        votes, raw_responses, evidence_list = [], [], []
        for choice in choices:
            raw = (choice.get("message") or {}).get("content") or ""
            raw = raw.strip()
            raw_responses.append(raw)
            verdict, evidence = self.parse_verdict(raw)
            votes.append(verdict)
            if evidence:
                evidence_list.append(evidence)

        while len(votes) < k_samples:
            votes.append("SUSPECT")
            raw_responses.append("")

        vote_counts = Counter(votes)
        majority_verdict, _ = vote_counts.most_common(1)[0]

        usage = body.get("usage") or {}
        return {
            "verdict": majority_verdict,
            "votes": dict(vote_counts),
            "evidence": evidence_list[0] if evidence_list else "",
            "raw_responses": raw_responses,
            "latency_ms": latency_ms,
            "prompt_tokens": usage.get("prompt_tokens") or 0,
            "model_id": model_id,
            "backend": "npu",
        }

    @staticmethod
    def parse_verdict(text: str) -> Tuple[str, str]:
        return parse_verdict(text)


class LocalLFMVerifierClient:
    """In-process PyTorch reference verifier (fallback / CPU-only servers)."""

    def __init__(self, model_id: str = DEFAULT_MODEL_ID):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.model_id = model_id
        self.backend = "local"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        self.model.eval()

    def evaluate_checkpoint(
        self,
        task_prompt: str,
        trajectory: str,
        prompt_template: str,
        k_samples: int = 3,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        formatted_prompt = prompt_template.format(
            task=task_prompt, trajectory=trajectory)
        messages = [{"role": "user", "content": formatted_prompt}]
        chat_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(chat_prompt, return_tensors="pt")
        n_in = inputs["input_ids"].shape[1]

        batch_inputs = {
            "input_ids": inputs["input_ids"].repeat(k_samples, 1),
            "attention_mask": inputs["attention_mask"].repeat(k_samples, 1),
        }

        t0 = time.time()
        votes, raw_responses, evidence_list = [], [], []
        with self.torch.no_grad():
            gen_out = self.model.generate(
                **batch_inputs,
                max_new_tokens=25,
                do_sample=(temperature > 0.0),
                temperature=max(temperature, 0.1),
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        for i in range(k_samples):
            raw = self.tokenizer.decode(gen_out[i][n_in:], skip_special_tokens=True).strip()
            raw_responses.append(raw)
            verdict, evidence = self.parse_verdict(raw)
            votes.append(verdict)
            if evidence:
                evidence_list.append(evidence)

        latency_ms = (time.time() - t0) * 1000
        vote_counts = Counter(votes)
        majority_verdict, _ = vote_counts.most_common(1)[0]

        return {
            "verdict": majority_verdict,
            "votes": dict(vote_counts),
            "evidence": evidence_list[0] if evidence_list else "",
            "raw_responses": raw_responses,
            "latency_ms": latency_ms,
            "prompt_tokens": n_in,
            "backend": "local",
        }

    @staticmethod
    def parse_verdict(text: str) -> Tuple[str, str]:
        return parse_verdict(text)


def LFMVerifierClient(
    model_id: str = None,
    npu_endpoint: str = None,
    npu_port: str = None,
) -> Any:
    backend = (_SETTINGS.get("VERIFIER_BACKEND") or "auto").lower()
    if npu_endpoint is not None:
        endpoint = npu_endpoint
    else:
        port = npu_port or _SETTINGS.get("LFM_NPU_PORT")
        base = _SETTINGS.get("LFM_NPU_ENDPOINT")
        if base.startswith("http://") and ":" in base.split("//")[1]:
            host, _, path = base.split("//", 1)[1].partition(":")
            host_only = "http://" + host.split(":")[0]
        else:
            base_host = "http://" + base.split("//")[-1]
            host_only = base_host.rstrip("/")
        if port:
            endpoint = f"{host_only}:{port}" + ("" if base.startswith("http://") else "/v1")
        else:
            endpoint = base
    wants_remote = False
    if model_id:
        wants_remote = False
    elif backend == "npu":
        wants_remote = True
    elif backend == "local":
        wants_remote = False
    else:
        print(f"[*] Probing NPU verifier endpoint: {endpoint}")
        wants_remote = NPUVerifierClient.healthy(endpoint)
        if wants_remote:
            print("[+] NPU verifier endpoint reachable — remote backend engaged.")
        else:
            print("[*] NPU endpoint unreachable — falling back to local verifier.")
    if wants_remote:
        return NPUVerifierClient(endpoint)
    model_id = model_id or _SETTINGS.get("LFM_MODEL_ID") or DEFAULT_MODEL_ID
    print(f"[*] Loading local verifier: {model_id}")
    return LocalLFMVerifierClient(model_id)


def parse_verdict(text: str) -> Tuple[str, str]:
    upper = text.upper()
    evidence = ""
    ev_match = re.search(r"<evidence>(.*?)</evidence>", text, re.DOTALL | re.IGNORECASE)
    if ev_match:
        evidence = ev_match.group(1).strip()
    elif '"' in text:
        quotes = re.findall(r'"([^"]*)"', text)
        if quotes:
            evidence = quotes[0].strip()

    if "ABORT" in upper:
        return "ABORT", evidence
    if "SUSPECT" in upper:
        return "SUSPECT", evidence
    return "CONTINUE", evidence
