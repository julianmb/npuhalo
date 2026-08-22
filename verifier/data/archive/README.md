# Set D Archive

## set_d_agentic_v1.jsonl
Version 1 (sha256 5e816fe5dcffd859… lineage). Superseded by version 2
(`../set_d_agentic.jsonl`, sha256 ee46b12d…).

Known defects fixed in v2:
- D04: instruction demanded *ignoring* missing-item restock while the hidden
  grader asserted the item was created (`items['zz']==3`) — contradictory.
- D06: instruction demanded preserving blank lines while the hidden grader's
  expected output omitted one; also ambiguous inline-comment semantics.
- D17: shipped casefile.py used `os.environ` without importing os.

All measurements taken against v1 (including every number in
verifier/results/) carry this caveat: absolute pass rates are deflated by
unsolvable tasks; arm-vs-arm comparisons remain internally valid.
