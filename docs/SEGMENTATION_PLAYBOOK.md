# Document Segmentation Playbook

## 1. Fast first-pass segmentation

| Method | How it works | When to use it |
| --- | --- | --- |
| **Paragraph / sentence split** | Regex or spaCy `sentencizer` → keep ≤ 3 sentences per chunk with 20 % token overlap | Quick ingest; you just need tokens < context window |
| **Token window (Recursive splitter)** | Fixed 400-token windows that recurse on punctuation; shipped in LangChain & Llama-Index | Baseline RAG that “just works” |

*Pros:* zero ML; sub-millisecond.
*Cons:* often cuts a logical idea in half and spams near-duplicates.

## 2. Semantic boundary detection

| Algorithm | Core idea | OSS/refs |
| --- | --- | --- |
| **TextTiling / lexical cohesion** | Embed each sentence → cosine sim drop < τ marks a topic break | `textsplit` (PyPI) |
| **CrossFormer (Ni et al., 2025)** | Transformer learns to spot subject shifts; 4 pp ↑ F1 vs. TextTiling on arXiv docs | |
| **Multi-granular splitter** | Recursively halves chunks until each sub-chunk passes a semantic-similarity test; keeps parent-child links for multi-resolution search | |

*Why it's better:* boundaries fall on real topic shifts, so retrieval returns fewer irrelevant neighbours.

## 3. LLM-assisted proposition extraction

| Flavor | Prompt sketch | Best for |
| --- | --- | --- |
| **Propositional / "Agentic" chunking** | "List the distinct factual statements in <text>; one clause per line." | Fact QA, reasoning tasks |
| **Triple extraction (S-P-O)** | "Convert each statement into subject / predicate / object triples." | Graph-style memory, deduping contradictions |
| **Paragraph-summary+bullet** | "Give a 1-sentence summary, then bullet the main claims." | When you still want a human-readable 'gist' for UI |

LLM cost is the bottleneck, so pipe batches through an 8 k-token context model or finetuned local model if volume is high.

## 4. Post-processing into belief chunks

1. **Hash for deduping** – SHA-256 of the normalised proposition text prevents storing the same idea twice.
2. **Attach metadata**

```jsonc
{
  "source_id": "...",      // doc+paragraph anchors
  "position": 17,          // ordinal inside doc
  "importance": salience_score(text),   // tf-idf or LLM
  "veracity":   source_confidence(url)  // or human label
}
```
3. **Embed** the proposition (MiniLM / E5) and **snap-assign** it to the nearest centroid (your “belief capsule”).
4. **Update the centroid & metadata** with your EMA rule (α≈0.05).

Result: each capsule now stores one drifting vector + merged metadata that **grows** as fresh evidence arrives.

## 5. Putting it together – reference pipeline (30 lines)

```python
for para in paragraphs(doc):
    # 1️⃣ semantic split inside the paragraph
    ideas = agentic_splitter(para)       # TextTiling → LLM if long

    for idea in ideas:
        if is_duplicate(idea): continue
        meta = build_meta(doc, idea)
        vec  = embed(idea)
        cid  = assign_centroid(vec)
        reconcile(vec, meta, cid)        # EMA + merge_meta from prior reply
```

Latency benchmarks on a single CPU (10 k ideas/min) come from using MiniLM embeddings and TextTiling first, calling the LLM only when segments > 512 tokens.

## 6. Checklist for production

* **Tune max-tokens per idea**: 60-120 tokens is the sweet spot for MiniLM / MTEB retrieval.
* **Keep overlap small (≤ 20 %)** to avoid duplicate centroids.
* **Evaluate**: Test answer F1 vs. chunk granularity; too coarse usually hurts long-tail factual recall, too fine inflates index & cost.
* **Monitor drift**: If a centroid’s intra-distance > δ, auto-split to keep coherency.

## Take-away

Break text incrementally:

1. **Cheap structural split ➜**
2. **Semantic boundary finder ➜**
3. **Optional LLM proposition extraction**

Then feed each atomic idea through your centroid-update “brain” logic.  This mirrors the hippocampus→cortex pipeline: raw episode → event boundary → gist proposition → integrated schema, giving you compact, self-updating beliefs ready for fast retrieval and reasoning.
