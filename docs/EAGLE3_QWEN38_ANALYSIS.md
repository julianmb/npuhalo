# Would an EAGLE-3 Draft-Model Trained *for* Qwen 3.8 27B Change the Outcome?

**Short answer: No — not for decode speed.** It would raise acceptance and τ,
which is necessary but not sufficient. The binding constraint on this machine
is bytes per decoded token on the shared LPDDR5X bus, and a better-trained
head is still a *separate* Drafter DRAM-streaming weights every chain step.
The numbers in this repo already prove it.

---

## 1. What "a Qwen-3.8-specific EAGLE-3" would actually fix

The draft-model used in all EAGLE-3 experiments here (`Ex0bit/Qwen3.6-27B-PRISM-EAGLE3`)
was trained against Qwen 3.6, not Qwen 3.8. That mismatch cost:

| Head | Acceptance | τ (tokens/verify) | Decode |
|---|---|---|---|
| Stock compressed d2t (32k vocab, Qwen-3.6-trained) | **7.4%** | 1.10 | 12.4 tok/s |
| Custom d2t, freq-derived, same weights | **90.5%** | 2.84 | 24.9 tok/s |
| Embedded MTP (trained inside Qwen 3.8 itself) | ~80% | ~2.5 | **33.8–41 tok/s** |

Two independent problems were interacting: (a) vocab coverage (the 32k d2t
vocab misses ~93% of Qwen 3.8's token distribution) and (b) distributional
drift between target versions. Fixing (a) with a frequency-derived d2t already
dragged acceptance from 7.4% to 90.5%. A Qwen-3.8-trained head would mainly
fix (b) — a **marginal** improvement over the 90.5% we already measured with
weights that were never trained for this target.

So the honest expected gain of a *perfectly* matched EAGLE-3 head is:
τ ≈ 2.9–3.1 instead of 2.84 (vs embedded MTP's ~2.5). Call it +2–10% on
the acceptance side, most plausibly landing near τ=3.0.

## 2. Why acceptance doesn't rescue the decode speed

Decode speed on this APU is bandwidth-limited, not compute-limited. The
byte-budget law (see `REPORT.md` §5):

```
tok/s ≈ ΣBW_streamed / bytes_per_decoded_token
```

Bare-mode calibration: the 13.55 GiB FP4 stream at 14.1 tok/s implies the
memory system sustains roughly **27.5 GB/s of full-model streaming throughput**
under the long sequential GEMV pattern typical of decode. Bytes per decoded
token under MTP: 13.55 GiB ÷ 2.5 (τ) ≈ 5.5 GB/model pass, plus 163 MB head
cost, divided over ~2.5 committed tokens.

| Drafter | Added cost | Ceiling |
|---|---|---|
| Embedded MTP K=4 | +163 MB (τ=2.5) | **~38–41 tok/s** ← matches measured |
| Custom EAGLE-3 Q4 (τ=2.84) | +367 MB/step, extra 1.77 GB per chain | ~30 tok/s theoretical |
| Qwen-3.8-trained EAGLE-3 Q4 (τ≈3.0) | +350 MB/step | ~30–32 tok/s theoretical |
| **Hypothetical SRAM-resident head (400 MB in MALL)** | 0 DRAM | ~parity vs MTP |

A trained-for-Qwen-3.8 head **tops out near 30–32 tok/s theory, vs the 33.8–41
tok/s we already achieve at 0% extra DRAM traffic**. Even if the drafter weights
waved their DRAM cost (4 MiB XDNA2 L2 cannot hold a 350 MB head; the 32 MB
"MALL" is Infinity Cache and holds the 27B model's KV + activations, not a
second drafter), an SRAM-resident head would land at roughly parity with
embedded MTP, not beyond it.

The reason is structural: embedded MTP's ~163 MB/token drift only conflicts
with 2 of the target's 38 verify steps. The head's 675 MB output GEMV is
essentially fused with the target's stream. EAGLE-3 re-materializes that
stream as a separate weight footprint every chain step, adding τ×(head cost)
bytes in the bank of the *sequential* drafting window that must complete
before verification starts.

## 3. Causality kills the overlap interpretation

The clean A/B/A contention test (final_verdict §6.1) showed concurrent NPU
generation cost the iGPU −16.5%. If EAGLE-3 drafting instead *"overlapped"*
window N+1 with verification of window N:

- P(A = q = 4), the full-window survival probability with p≈0.90: **65.6%**.
- 34.4% of ahead drafts are wasted and must be re-drafted → exposed NPU drafter
  streaming after rejection.
- The hidden state for window N+1 does not exist until verification finishes.
  Drafting from stale state lowers acceptance further.
- Rollback, ring-buffer sync, and slot-seek all add overhead `O` on top.

Substituting into `V_conc + (1-s)·D + O < L·T1` (final_verdict §6.1 formula
set) with q=4, s=0.656, V_conc = verification slowed by concurrent DRAM
traffic → the inequality does not hold at any measured parameter combination.

## 4. The only theoretical win is not a better head

The one architecture that could beat embedded MTP on this bus is *parallel
target-aligned drafting* (PARD-2 class), where the drafter proposes chains of
length proportional to τ without the causal dependency on the target's
per-step hidden state, and its weight stream is amortized accordingly. Nothing
is trained or shipped for it (see final_verdict §2), and a Qwen-3.8-trained
version of Ex0bit's EAGLE-3 head does not fix this either: same serial
chain, same DRAM stream per chain step, same hidden-state causality.

A better-trained head raises τ — but measured τ is already near saturation
(2.84–2.88 out of the ~3.5–4.0 practical ceiling for this head family). What
limits decode is the DRAM bytes the chain itself costs, not the model's skill
at predicting tokens.

## 5. What would actually change the picture

| Path | Ceiling | Blocking dependency |
|---|---|---|
| Qwen-3.8-trained EAGLE-3 | ~30–32 tok/s | training + still loses to embedded MTP by design |
| **PARD-2 style parallel drafting** (τ→8) | **~56 tok/s** | PACE runtime + intention-trained weights; nothing shipped |
| SRAM-resident EAGLE (≤4 MB head) | ~parity vs MTP | impossible at any useful vocabulary |
| DRAM bus upgrade / Q4→Q3 target quant | — | hardware / quality tradeoff |

**Bottom line:** a Qwen-3.8-specific EAGLE-3 head is a nice-to-have for
acceptance sensitivity curves and vocabulary research, but it cannot raise
this platform's decode ceiling — the bus, not the Drafter's skill, is the
bottleneck.
