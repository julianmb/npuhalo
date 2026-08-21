# Phase 1: Repository Integration & API Specification Analysis

**Target Repository:** [`llm-as-a-verifier/llm-as-a-verifier`](https://github.com/llm-as-a-verifier/llm-as-a-verifier)  
**Analysis Date:** 2026-08-20  
**Analyzed Module:** `llm_verifier/fine_grained_reward.py`, `llm_verifier/prompts.py`, `llm_verifier/pivot_tournament.py`

---

## 1. Expected API Interface & Endpoints

`llm-as-a-verifier` interacts with inference backends through standard client abstractions:
* **Primary Endpoint:** OpenAI-compatible `/v1/chat/completions` via the `openai` Python SDK (`create_openai_client`, `create_deepseek_client`), or Google GenAI SDK (`create_gemini_client` via Vertex AI).
* **Interface Method:** `client.chat.completions.create(...)`
* **Streaming vs Batch:** Synchronous non-streaming completions executed across thread pools (`ThreadPoolExecutor` with 50–500 workers).

---

## 2. Request Fields & Parameters

Referenced in `llm_verifier/fine_grained_reward.py:L417-L432`:

```python
params = dict(
    model=resolve_model(client, model),
    messages=[{"role": "user", "content": content}],
    max_tokens=4096,
    temperature=1.0,
    logprobs=True,
    top_logprobs=top_logprobs,  # standard G=20
)
```

### Key Request Fields:
1. `logprobs: bool = True` — Mandates token logprobabilities in the response.
2. `top_logprobs: int = 20` — Requests the top-20 candidate tokens at each position.
3. `temperature: float = 1.0` — Preserves the raw calibrated logit distribution.
4. `max_tokens: int = 4096` (or 32768 for reasoning models) — Leaves sufficient budget for CoT reasoning before score tags.
5. `extra_body` options (for vLLM/SGLang):
   * `{"chat_template_kwargs": {"enable_thinking": False}}` or `structured_outputs` for prefilled evaluation.

---

## 3. Response Structure & Token Parsing

Referenced in `llm_verifier/fine_grained_reward.py:L439-L448`:

The verifier extracts token distributions from `response.choices[0].logprobs.content`:
```python
for pos in choice.logprobs.content:
    tokens.append(pos.token)
    alts = [(alt.token, alt.logprob) for alt in (pos.top_logprobs or [])]
    position_logprobs.append(alts)
```

Each position $i$ contains:
* `pos.token`: The sampled token string.
* `pos.top_logprobs`: List of alternative `(token_str, logprob)` pairs.

---

## 4. Score-Token Vocabulary & Mathematical Formulation

Referenced in `llm_verifier/fine_grained_reward.py:L69-L90` and `llm_verifier/fine_grained_reward.py:L663-L687`:

* **Granularity ($G=20$):** 20 discrete rating letters (`A` through `T`, case-insensitive).
  * `A` = best (value = 20.0)
  * `B`..`S` = intermediate (values 19.0 down to 2.0)
  * `T` = worst (value = 1.0)

### Continuous Expectation Formula:
Given the logprob distribution at the token position immediately following `<score_A>` or `<score_B>`:
$$P(v_g) = \exp(\text{logprob}(v_g))$$
$$\text{Total } P = \sum_{g=1}^{G} P(v_g)$$
$$\mathbb{E}[\text{Raw Score}] = \frac{\sum_{g=1}^{G} \phi(v_g) \cdot P(v_g)}{\text{Total } P}$$
$$\text{Normalized Score } R = \frac{\mathbb{E}[\text{Raw Score}] - 1.0}{20.0 - 1.0} \in [0.0, 1.0]$$

---

## 5. Tokenization Constraints & Tag Matching

Referenced in `llm_verifier/fine_grained_reward.py:L636-L660`:

* **Tag Matching (`_find_tag_logprobs`):** Searches tokens for closing tags `<score_A>` / `<score_B>` (or fused forms like `<score_A` or `>A`).
* **Position Offset:** Reads logprobs at position $i+1$ directly following the tag.
* **Token Fusion Handling:** Strips leading spaces and closing brackets (`>` prefix) to robustly identify tokens across different byte-pair encodings (SentencePiece, Tiktoken, BPE).
* **Prefill Strategy (`_score_tags_by_prefill`):** For open-source models that struggle with strict tag emission, the repo supports prefilling `assistant` message with `\n<score_X>` and sampling 1 token with `structured_outputs: {"choice": letters}`.

---

## 6. Summary of Architectural Requirements for Local Backends

| Requirement | Description | Criticality |
| :--- | :--- | :--- |
| **Endpoint** | `/v1/chat/completions` (OpenAI format) | **Hard Requirement** |
| **Logprobs Output** | `choice.logprobs.content[].top_logprobs` with $\ge 20$ alternatives | **Hard Requirement** |
| **Token Budget** | Must support reasoning traces ($\ge 2048$ tokens) before verdict | **Medium** |
| **Score Distribution** | Competing probabilities over `A`–`T` tokens | **Hard Requirement** |
| **Context Length** | $\ge 8\text{k} - 32\text{k}$ for trajectory evaluation | **High** |
