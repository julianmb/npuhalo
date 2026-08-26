# DeepSeek-V4-Pro Production Config & Modeling Audit

**Target Model:** `deepseek-ai/DeepSeek-V4-Pro` / `deepseek-ai/DeepSeek-V4-Pro-0813`  
**Hugging Face Source:** `https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro`  
**Inspected Artifacts:**
- `config.json`
- `inference/config.json`
- `inference/model.py` (lines 1–827)
- `encoding/encoding_dsv4.py`

---

## 1. Summary of Discovered Configuration Fields

The official `DeepSeek-V4-Pro` configuration contains explicit architectural fields corresponding to static lookup memory, sparse index attention, and manifold hyper-connections:

| Field Name (`config.json`) | Type / Value | Corresponding Module (`inference/model.py`) | Architectural Role |
| :--- | :--- | :--- | :--- |
| `num_hash_layers` / `n_hash_layers` | `int = 3` | `Gate` (lines 546–585) | **Deterministic Token-to-Expert static lookup table** for the first 3 MoE layers (0, 1, 2). |
| `index_n_heads` | `int = 64` | `Indexer` (lines 380–434) | Number of sparse attention indexing heads. |
| `index_head_dim` | `int = 128` | `Indexer` (lines 380–434) | Dimension per indexing head for compressed KV scoring. |
| `index_topk` | `int = 1024` / `512` | `Indexer` (lines 380–434) | Top-K compressed positions selected per query token. |
| `hc_mult` | `int = 4` | `HyperConnections` (lines 648–702) | Number of parallel residual stream branches ($M=4$). |
| `hc_sinkhorn_iters` | `int = 20` | `hc_split_sinkhorn` (line 679) | Iterations for doubly-stochastic Sinkhorn routing matrix normalization. |
| `hc_eps` | `float = 1e-06` | `HyperConnections` (lines 662, 710) | Epsilon for hyper-connection numerical stability. |
| `num_hidden_layers` | `int = 61` | `Transformer` | Total backbone transformer blocks. |

---

## 2. Granular Code Analysis: Where and How They Function

### 2.1 Static Token-to-Expert Lookup (`num_hash_layers = 3`)
- **Location:** `inference/model.py:546-585` (`Gate`)
- **Code implementation:**
  ```python
  class Gate(nn.Module):
      def __init__(self, layer_id: int, args: ModelArgs):
          super().__init__()
          self.dim = args.dim
          self.topk = args.n_activated_experts
          self.score_func = args.score_func
          self.route_scale = args.route_scale
          self.hash = layer_id < args.n_hash_layers  # Active for layers 0, 1, 2
          self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
          if self.hash:
              self.tid2eid = nn.Parameter(
                  torch.empty(args.vocab_size, args.n_activated_experts, dtype=torch.int32), 
                  requires_grad=False
              )
              self.bias = None
          else:
              self.bias = nn.Parameter(torch.empty(args.n_routed_experts, dtype=torch.float32))

      def forward(self, x: torch.Tensor, input_ids: Optional[torch.Tensor] = None):
          scores = linear(x.float(), self.weight.float())
          ...
          if self.hash:
              indices = self.tid2eid[input_ids]   # <-- O(1) STATIC DIRECT LOOKUP
          else:
              indices = scores.topk(self.topk, dim=-1)[1]
          weights = original_scores.gather(1, indices)
          ...
          return weights, indices
  ```
- **Significance:** In the first 3 layers, neural top-K routing (`scores.topk()`) is **completely bypassed**. Instead, expert routing indices are indexed directly via a static lookup table `tid2eid[input_ids]`. This mirrors the Engram principle: offloading static token associations to $\mathcal{O}(1)$ lookup tables so early layers avoid dynamic computation overhead.

### 2.2 Compressed KV Attention Indexer (`index_n_heads`, `index_head_dim`, `index_topk`)
- **Location:** `inference/model.py:380-434` (`Indexer`)
- **Code implementation:**
  ```python
  class Indexer(torch.nn.Module):
      """Selects top-k compressed KV positions for sparse attention via learned scoring."""
      def __init__(self, args: ModelArgs, compress_ratio: int = 4):
          ...
          self.n_heads = args.index_n_heads       # 64
          self.head_dim = args.index_head_dim     # 128
          self.index_topk = args.index_topk       # 1024
          ...
          self.compressor = Compressor(args, compress_ratio, self.head_dim, True)
  ```
- **Significance:** Implements DeepSeek's sparse attention indexing mechanism, selecting 1,024 compressed KV slots from up to 1M context positions.

### 2.3 Manifold Hyper-Connections (`hc_mult = 4`)
- **Location:** `inference/model.py:648-702` (`HyperConnections`)
- **Significance:** Directly matches the $M=4$ hyper-connection multi-branch residual stream topology formulated in Section 2.4 of the Engram paper.

---

## 3. Production Confirmation vs. Research Repo

1. **Active in Shipped Production Configs:** The presence of `num_hash_layers: 3` and `tid2eid` in official `DeepSeek-V4-Pro` production configs confirms that DeepSeek has graduated static lookup memory from a research concept to standard production architecture.
2. **Evolution from Standalone Engram:**
   - In the standalone Engram paper, lookup memory is stored in high-dimensional continuous embedding tables ($d_{\text{mem}} = 1024$, multi-head $K=8$).
   - In `DeepSeek-V4-Pro`, static lookup is integrated directly into the MoE gate as an integer table mapping tokens directly to expert indices (`tid2eid`), while `Indexer` handles sequence-level sparse retrieval.
3. **Implications for Qwen3.8-Flash-Next:**
   Rumors of Qwen3.8-Flash-Next featuring a 51B-parameter N-gram lookup table represent the full-scale instantiation of the Engram continuous embedding architecture (combining massive static embedding tables with MoE computation).
