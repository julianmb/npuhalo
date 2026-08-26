# Reference Architecture Analysis: DeepSeek Engram

**Source Repository:** `github.com/deepseek-ai/Engram`  
**Reference Paper:** *Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models* (Cheng et al., DeepSeek-AI / Peking University, 2025)  
**Primary Reference Files:**
- `engram_demo_v1.py` (lines 1–423)
- `Engram_paper.pdf` (Sections 2.1–2.5, Appendix B–D)

---

## 1. N-Gram Hashing & Addressing Mechanics

Engram replaces dynamic MoE neural routing in designated layers with static $\mathcal{O}(1)$ multi-head hash table lookups over local token contexts.

```
Raw Token IDs: x_t 
       │
       ▼  (Normalizers: NFKC, NFD, StripAccents, Lowercase, Whitespace Collapse)
Canonical Token IDs: x'_t  [Vocabulary Compression: -23% slots]
       │
       ▼  (Base Shifts: [x'_t, x'_{t-1}, x'_{t-2}, ...])
Multi-Head Polynomial-XOR Hash Engine:
  For n-gram order n ∈ [2, N]:
    h_mix = (x'_t * M_0) ⊕ (x'_{t-1} * M_1) ⊕ ... ⊕ (x'_{t-n+1} * M_{n-1})
    For head k ∈ [0, K-1]:
      Slot_idx[n, k] = h_mix mod Prime_Modulus[n, k]
       │
       ▼
Concatenated Table Lookup: e_t = [E_2,0[idx], E_2,1[idx], ..., E_N,K-1[idx]]
```

### 1.1 Tokenizer Compression Layer
- **Source Reference:** `engram_demo_v1.py:60-122` (`CompressedTokenizer`), Paper Section 2.2 & Appendix C.
- **Problem Addressed:** Standard BPE/Byte-fallback tokenizers assign distinct token IDs to semantically identical words due to capitalization, whitespace prefixes, or Unicode variance (e.g., `" Apple"`, `"apple"`, `"APPLE"`).
- **Mechanism:** A pre-computed surjective mapping $\mathcal{P}: V \to V'$ normalizes tokens using `tokenizers.normalizers`:
  - NFKC / NFD normalization
  - `StripAccents()`
  - `Lowercase()`
  - Whitespace collapsing (`Regex(r"[ \t\r\n]+") -> " "`)
- **Impact:** Reduces effective vocabulary size by **~23%** (e.g., from 129,280 down to ~99,500 canonical IDs), significantly increasing the semantic hit rate and density of N-gram memory slots.

### 1.2 Hash Function Algorithm & Width
- **Source Reference:** `engram_demo_v1.py:188-297` (`NgramHashMapping`), Paper Section 2.2 Eq. (1).
- **Hash Width:** 64-bit unsigned/signed integers (`np.int64`).
- **Multiplier Generation:**
  - Layer-dependent seed: $\text{seed}_{\ell} = \text{seed}_{\text{base}} + 10007 \times \ell$ (`engram_demo_v1.py:222`).
  - Multipliers $M_k$ are random odd 64-bit integers: $M_k = 2 \cdot r_k + 1$, where $r_k \in [0, \frac{2^{63}-1}{2 \cdot |V'|}]$.
- **Bitwise Mixing:** For an $n$-gram $(x'_t, x'_{t-1}, \dots, x'_{t-n+1})$:
  $$\text{mix} = (x'_t \cdot M_0) \oplus (x'_{t-1} \cdot M_1) \oplus \dots \oplus (x'_{t-n+1} \cdot M_{n-1})$$
- **Multi-Head Modulo Mapping:**
  $$z_{t,n,k} = \text{mix} \pmod{M_{n,k}}$$
  where $M_{n,k}$ is a unique prime number calculated via `find_next_prime()` (`engram_demo_v1.py:181-186`) starting from the base N-gram vocabulary size.

### 1.3 Collision Handling Strategy
1. **Multi-Head Prime-Modulus Orthogonality:** Each N-gram order $n$ uses $K$ independent heads ($K=8$ default), each indexed modulo a distinct prime. Two distinct phrases that collide in head $k_1$ ($z_{t,n,k_1} = z_{t',n,k_1}$) will not collide in head $k_2$ due to coprime modular arithmetic ($\gcd(M_{n,k_1}, M_{n,k_2}) = 1$).
2. **Context-Aware Gating Suppression:** If a hash collision retrieves an irrelevant memory vector, the downstream query-key gating scalar $\alpha_t \to 0$, dynamically suppressing the noise before it enters the residual stream (`engram_demo_v1.py:365-376`).

---

## 2. Table Storage & Memory Format

- **Source Reference:** `engram_demo_v1.py:305-325` (`MultiHeadEmbedding`), Paper Section 2.2 Eq. (2).

### 2.1 Storage Layout
- Implemented as a single flat `nn.Embedding(num_embeddings=total_N, embedding_dim=D)` with an `offsets` buffer:
  $$\text{offsets} = [0, M_0, M_0 + M_1, \dots, \sum_{i=0}^{H-2} M_i]$$
  $$\text{total\_N} = \sum_{n=2}^N \sum_{k=1}^K M_{n,k}$$
- **Entry Dtype:** Standard `torch.bfloat16` or `torch.float32`. In quantized production deployment, sub-byte representations (INT8, FP8, INT4) are possible.
- **Sparsity Type:** Dynamic sparse lookup ($\mathcal{O}(1)$ gather per token). The underlying weight matrix is stored dense in memory.

### 2.2 Dimension & Capacity Hierarchy
- Per-head embedding dimension: $D = d_{\text{embed\_per\_ngram}} / K$ (e.g., $512 / 8 = 64$).
- Retrieved vector dimension: $d_{\text{mem}} = (N_{\max} - 1) \times d_{\text{embed\_per\_ngram}}$ (e.g., $(3 - 1) \times 512 = 1024$).
- **Reference Example Configurations (`engram_demo_v1.py:39-49`):**
  - `engram_vocab_size = [129280 * 5, 129280 * 5]` (~646K slots per N-gram order)
  - $N_{\max} = 3$ (evaluating 2-grams and 3-grams)
  - $K = 8$ heads per N-gram order (16 heads total per Engram layer)
  - `layer_ids = [1, 15]` (2 Engram layers in a 30-layer model)
  - Paper scales from $10^6$ up to $10^{11}$ slots (1M to 100B parameters).

---

## 3. Residual Stream Injection Architecture

- **Source Reference:** `engram_demo_v1.py:326-378` (`Engram`), Paper Section 2.3 & 2.4.

```
Retrieved Embedding e_t  [B, L, d_mem]
       │
       ├───► Value Proj W_V ───► v_t [B, L, D_hidden]
       │                              │
       └───► Key Projs W_K^(m)        ▼
               │ (m=1..M)        Contextual Gate: α_t^(m) = σ( (Norm(k_t^(m)) · Norm(q_t^(m))) / √d )
               ▼                      │
         k_t^(m)                      ▼
                           u_t^(m) = α_t^(m) · v_t
                                      │
                                      ▼
                        ShortConv1D (kernel=4, dilation=max_ngram, SiLU)
                                      │
                                      ▼
                  Injected into Backbone: h_t^(m) = h_t^(m) + u_t^(m)
```

### 3.1 Layer Placement
- Embedded in specific intermediate layers (e.g., layers 1 and 15 out of 30 in the 27B model).
- **Rationale (Paper Section 3 & Section 6.2):** Placing Engram in early/middle layers offloads static N-gram pattern reconstruction from early attention heads, preserving effective network depth for complex compositional reasoning.

### 3.2 Gating & Multi-Branch Hyper-Connections (mHC)
- Engram is integrated with Manifold-Constrained Hyper-Connections ($M=4$ parallel residual streams).
- **Shared Value Projection:** A single shared $W_V \in \mathbb{R}^{d_{\text{mem}} \times d_{\text{hidden}}}$ projects retrieved embeddings to hidden dimension.
- **Branch-Specific Key Projections:** $M$ distinct linear projections $\{W_K^{(m)}\}_{m=1}^M \in \mathbb{R}^{d_{\text{mem}} \times d_{\text{hidden}}}$.
- **Gating Equation:**
  $$\alpha_t^{(m)} = \sigma\left( \frac{\text{RMSNorm}(h_t^{(m)})^\top \text{RMSNorm}(W_K^{(m)} e_t)}{\sqrt{d}} \right)$$
- **Short 1D Convolution:**
  Gated representations $v_t$ pass through a causal depthwise 1D convolution (`ShortConv`, `kernel_size=4`, `dilation=max_ngram_size`, `SiLU` activation) to capture short-range temporal transitions before residual addition.

---

## 4. Modularity & Backend Decoupling

### 4.1 Dependency Decoupling Analysis
| Pipeline Stage | Inputs | Dependencies | Offload Feasibility |
| :--- | :--- | :--- | :--- |
| **1. Tokenizer Normalization** | Raw `input_ids` | CPU / Host Tokenizer lookup | **100% Async / Host** |
| **2. N-Gram Multi-Head Hash** | Normalized `input_ids` | Deterministic bitwise XOR/mul | **100% Async / Host / NPU** |
| **3. Table Gather (Lookup)** | Hash indices $z_{t,n,k}$ | Large embedding tables $E_{n,k}$ | **100% Async / Host RAM / NPU** |
| **4. Key/Value Projections & Gate** | Retrieved $e_t$ + Hidden $h_t$ | Layer $L$ intermediate states | GPU Compute bound |
| **5. ShortConv & Residual Add** | Gated $u_t$ + Hidden $h_t$ | Layer $L$ intermediate states | GPU Compute bound |

### 4.2 Key System Implication for NPU Offload
Because stages 1, 2, and 3 depend **exclusively on token IDs** ($x_{1:t}$) and are completely independent of intermediate transformer activations ($h_t^{(\ell)}$), the entire hash + gather pipeline can be executed **asynchronously ahead of time**.

While the GPU is computing Layer 0, the NPU/Host RAM can prefetch and gather the embeddings $e_t$ for Layer 1. While the GPU computes Layers 2 through 14, the lookup for Layer 15 can be completely hidden across PCIe.

---

## 5. Codebase Infrastructure & Pretrained Checkpoints

- **Quantization Support:** None in reference demo repository (standard PyTorch `nn.Embedding`).
- **ONNX Export / C++ Kernels:** None provided in `deepseek-ai/Engram`.
- **Pretrained Checkpoints:** No weight checkpoints are hosted in the `deepseek-ai/Engram` repository. All weights must be simulated or derived from production models.
