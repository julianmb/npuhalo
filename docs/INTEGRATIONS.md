# Agent & Tool Integrations Guide

`npuhalo-proxy` exposes a standard OpenAI-compatible API at `http://localhost:8000/v1`. You can plug it into any local coding assistant, IDE extension, or agent framework.

---

## 1. Aider

[Aider](https://github.com/paul-gauthier/aider) is a leading CLI AI pair programmer. You can route Aider through `npuhalo` to enjoy fast NPU routing and zero-latency safety auditing:

```bash
aider --openai-api-base http://localhost:8000/v1 \
      --openai-api-key none \
      --model openai/npuhalo
```

Or add to your `~/.aider.conf.yml`:
```yaml
openai-api-base: http://localhost:8000/v1
openai-api-key: none
model: openai/npuhalo
```

---

## 2. Claude Code & OpenCodeInterpreter

To point CLI agents using OpenAI-compatible endpoints to `npuhalo`:

```bash
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="none"
```

---

## 3. Continue.dev (VS Code & JetBrains)

In your `~/.continue/config.json`, add `npuhalo` under `models`:

```json
{
  "models": [
    {
      "title": "npuhalo (AMD Strix Halo / Point)",
      "provider": "openai",
      "model": "npuhalo",
      "apiBase": "http://localhost:8000/v1",
      "apiKey": "none"
    }
  ]
}
```

---

## 4. OpenWebUI / LibreChat

1. Open **Settings** $\to$ **Connections** $\to$ **OpenAI API**.
2. Set API URL: `http://localhost:8000/v1`
3. Set API Key: `none`
4. Click **Verify Connection**. All models served via the proxy (`Ornith-1.5-35B-A3B` and `minicpm5:2b`) will populate automatically.

---

## 5. Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="none",
)

response = client.chat.completions.create(
    model="npuhalo",
    messages=[
        {"role": "user", "content": "What is the capital of Japan?"}
    ],
)
print(response.choices[0].message.content)
```

---

## 6. cURL

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```
