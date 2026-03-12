# Issues / Risks (append-only)

- Eval `hit_doc/hit_page` may be mis-measured due to doc-name normalization mismatches (punctuation, aliasing).
- `answer_preview` truncation can hide citations/references, so answer-quality assessment from JSONL is unreliable.
- Retrieved doc list is truncated (top10 ids only) and may include duplicates; cannot distinguish doc-level vs chunk-level duplication without richer logging.
- Current doc hit check still allows normalized substring match (gold in candidate) for tolerance, so very short/ambiguous gold doc names can over-match; consider exact-vs-substring policy tuning if false positives appear.
