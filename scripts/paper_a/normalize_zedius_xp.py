"""
Task 2: Normalize ZEDIUS_XP -> SUPRA_XP in document_scope_table.csv and shared_doc_gold.csv.
Only device_name_norm and candidate_devices columns are modified. device_name_raw is untouched.
"""
import csv
import json
import os

DATA_DIR = "/home/hskim/work/llm-agent-v2/data/paper_a/corpus_labels"
EVIDENCE_DIR = "/home/hskim/work/llm-agent-v2/.sisyphus/evidence/paper-a/reports"

SCOPE_TABLE = os.path.join(DATA_DIR, "document_scope_table.csv")
SHARED_GOLD = os.path.join(DATA_DIR, "shared_doc_gold.csv")
AUDIT_OUT = os.path.join(EVIDENCE_DIR, "T1_scope_table_alias_audit_2026-03-09.json")

os.makedirs(EVIDENCE_DIR, exist_ok=True)

# --- document_scope_table.csv ---
with open(SCOPE_TABLE, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = list(reader)

before_count = sum(1 for r in rows if r["device_name_norm"] == "ZEDIUS_XP")
changed_doc_ids = [r["doc_id"] for r in rows if r["device_name_norm"] == "ZEDIUS_XP"]

for r in rows:
    if r["device_name_norm"] == "ZEDIUS_XP":
        r["device_name_norm"] = "SUPRA_XP"

after_count = sum(1 for r in rows if r["device_name_norm"] == "ZEDIUS_XP")

with open(SCOPE_TABLE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"document_scope_table.csv: {before_count} -> {after_count} ZEDIUS_XP rows")

# --- shared_doc_gold.csv ---
with open(SHARED_GOLD, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    gold_fieldnames = reader.fieldnames
    gold_rows = list(reader)

gold_norm_before = sum(1 for r in gold_rows if r.get("device_name_norm", "") == "ZEDIUS_XP")
gold_cand_before = sum(1 for r in gold_rows if "ZEDIUS_XP" in r.get("candidate_devices", ""))

changed_gold_ids = []
for r in gold_rows:
    changed = False
    if r.get("device_name_norm", "") == "ZEDIUS_XP":
        r["device_name_norm"] = "SUPRA_XP"
        changed = True
    if "ZEDIUS_XP" in r.get("candidate_devices", ""):
        r["candidate_devices"] = r["candidate_devices"].replace("ZEDIUS_XP", "SUPRA_XP")
        changed = True
    if changed:
        changed_gold_ids.append(r["doc_id"])

gold_norm_after = sum(1 for r in gold_rows if r.get("device_name_norm", "") == "ZEDIUS_XP")
gold_cand_after = sum(1 for r in gold_rows if "ZEDIUS_XP" in r.get("candidate_devices", ""))

with open(SHARED_GOLD, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=gold_fieldnames)
    writer.writeheader()
    writer.writerows(gold_rows)

print(f"shared_doc_gold.csv device_name_norm: {gold_norm_before} -> {gold_norm_after}")
print(f"shared_doc_gold.csv candidate_devices: {gold_cand_before} -> {gold_cand_after}")

# --- Audit JSON ---
audit = {
    "date": "2026-03-09",
    "task": "T2_normalize_ZEDIUS_XP_to_SUPRA_XP",
    "document_scope_table": {
        "before_count": before_count,
        "after_count": after_count,
        "changed_doc_ids": changed_doc_ids,
    },
    "shared_doc_gold": {
        "device_name_norm_before": gold_norm_before,
        "device_name_norm_after": gold_norm_after,
        "candidate_devices_before": gold_cand_before,
        "candidate_devices_after": gold_cand_after,
        "changed_doc_ids": changed_gold_ids,
    },
}

with open(AUDIT_OUT, "w", encoding="utf-8") as f:
    json.dump(audit, f, indent=2, ensure_ascii=False)

print(f"Audit written to: {AUDIT_OUT}")
