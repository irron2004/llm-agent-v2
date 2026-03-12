# Cross-Device Topic Feasibility for Counterfactual Scope Traps

Date: 2026-03-12
Status: Measured from current Paper A corpus artifacts

## 1) Question

Can we construct counterfactual scope-trap queries (same topic, wrong device) from the current corpus at sufficient scale?

## 2) Data Sources

- `.sisyphus/evidence/paper-a/corpus/doc_meta.jsonl`
- `.sisyphus/evidence/paper-a/policy/shared_topics.json`
- `scripts/paper_a/build_shared_and_scope.py`

Policy reference:
- `build_shared_and_scope.py` defines shared-topic eligibility from SOP-only topic-device graph and marks `is_shared = (deg >= 3)`.

## 3) Measurement Method

We computed topic overlap in two grouping modes:

1. Raw topic string grouping
2. Normalized topic grouping (lowercase + non-alnum collapsed to spaces)

Definitions:
- cross-device topic: appears in >= 2 distinct devices
- trap-ready topic: cross-device topic where each involved device has at least one SOP doc (`manifest_doc_type in {sop_pdf, sop_pptx}`)

## 4) Results

Corpus base:
- total rows: 578
- rows with missing topic: 14

Raw topic grouping:
- unique topics: 418
- cross-device topics (>=2 devices): 57
- topics with >=3 / >=4 / >=5 devices: 14 / 5 / 2
- trap-ready topics: 52

Normalized topic grouping:
- unique topics: 366
- cross-device topics (>=2 devices): 74
- topics with >=3 / >=4 / >=5 devices: 30 / 14 / 5
- trap-ready topics: 68

Shared policy artifact:
- `shared_topics.json` keys: 374
- `is_shared=true`: 13

Interpretation:
- Cross-device overlap exists at practical scale in both raw and normalized views.
- Current `is_shared=true` count (13) is narrower than full cross-device overlap because policy requires stronger criteria (`deg >= 3` on SOP topic-device graph).
- Counterfactual scope traps are feasible now without synthetic topic invention.

## 5) High-Value Trap Candidate Topics (Normalized View)

Examples with strong multi-device overlap:
- `controller` (6 devices)
- `ffu` (6 devices)
- `device net board` (5 devices)
- `robot` (5 devices)
- `gas spring` (5 devices)
- `solenoid valve` (4 devices)
- `heater chuck` (4 devices)
- `ctc` (4 devices)
- `pressure switch` (4 devices)
- `slot valve` (3 devices)
- `flow switch` (3 devices)
- `sensor board` (3 devices)

These are suitable seeds for counterfactual pairs:
- in-scope variant: target device matches query intent
- trap variant: same topic but wrong device/family document injected into candidates

## 6) Reproducibility Command

```bash
python - <<'PY'
import json, re
from collections import defaultdict
from pathlib import Path

rows=[json.loads(x) for x in Path('.sisyphus/evidence/paper-a/corpus/doc_meta.jsonl').read_text().splitlines() if x.strip()]
SOP_TYPES={'sop_pdf','sop_pptx'}

def norm_topic(t: str) -> str:
    t=t.lower().strip()
    t=re.sub(r'[^a-z0-9]+',' ',t)
    return ' '.join(t.split())

def calc(use_norm: bool):
    topic_devices=defaultdict(set)
    topic_docs=defaultdict(list)
    topic_device_has_sop=defaultdict(lambda: defaultdict(bool))
    for r in rows:
        topic=(r.get('topic') or '').strip()
        dev=(r.get('es_device_name') or '').strip()
        if not topic or not dev:
            continue
        k=norm_topic(topic) if use_norm else topic
        topic_devices[k].add(dev)
        topic_docs[k].append(r.get('es_doc_id',''))
        if (r.get('manifest_doc_type') or '').lower() in SOP_TYPES:
            topic_device_has_sop[k][dev]=True
    cross={k:v for k,v in topic_devices.items() if len(v)>=2}
    trap_ready=[k for k,devs in cross.items() if all(topic_device_has_sop[k].get(d,False) for d in devs)]
    return {
        'unique_topics': len(topic_devices),
        'cross_device_topics': len(cross),
        'topics_ge3_devices': sum(1 for v in topic_devices.values() if len(v)>=3),
        'topics_ge4_devices': sum(1 for v in topic_devices.values() if len(v)>=4),
        'topics_ge5_devices': sum(1 for v in topic_devices.values() if len(v)>=5),
        'trap_ready_topics': len(trap_ready),
    }

shared=json.loads(Path('.sisyphus/evidence/paper-a/policy/shared_topics.json').read_text())
print(json.dumps({
    'rows_total': len(rows),
    'missing_topic_rows': sum(1 for r in rows if not (r.get('topic') or '').strip()),
    'raw': calc(False),
    'normalized': calc(True),
    'shared_topics_json_total': len(shared),
    'shared_topics_json_is_shared_true': sum(1 for v in shared.values() if v.get('is_shared')),
}, indent=2))
PY
```

## 7) Immediate Next Step

Generate a first trap set by sampling from normalized trap-ready topics:
- sample policy: 10 topics x 3 device pairs x 2 query variants (in-scope + trap)
- total initial set: 60 queries
- keep non-seeded wording ratio >= 30%

## 8) Generated Artifact

- `.sisyphus/evidence/paper-a/corpus/cross_device_trap_candidates.json`
  - normalized cross-device topic inventory
  - per-topic device pairs
  - per-device candidate `doc_id` list
  - `trap_ready` flag for direct counterfactual construction
