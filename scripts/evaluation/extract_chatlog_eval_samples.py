"""Extract diverse evaluation samples from reference service chat logs.

최근 3개월(`data/chat_logs/monthly/chat_logs_2026-0{1,2,3}.txt`)에서
질문 유형이 골고루 분포하도록 20개의 대표 샘플을 뽑아 JSON + Markdown 으로 저장한다.

산출물:
- docs/tasks/TASK-20260421-react-agent-chatlog-alignment/samples/samples.json
- docs/tasks/TASK-20260421-react-agent-chatlog-alignment/samples/samples.md
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
LOG_FILES = [
    REPO / "data/chat_logs/monthly/chat_logs_2026-01.txt",
    REPO / "data/chat_logs/monthly/chat_logs_2026-02.txt",
    REPO / "data/chat_logs/monthly/chat_logs_2026-03.txt",
]
OUT_DIR = REPO / "docs/tasks/TASK-20260421-react-agent-chatlog-alignment/samples"
SEP = "=" * 50

# PII 마스킹용
PII_PATTERNS = [
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}"), "[EMAIL]"),
    (re.compile(r"문의: [^\n(]+\("), "문의: [NAME] ("),
]


def mask_pii(text: str) -> str:
    for pat, repl in PII_PATTERNS:
        text = pat.sub(repl, text)
    return text


def parse_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    blocks = text.split(SEP)
    out: list[dict[str, Any]] = []
    for b in blocks:
        b = b.strip()
        if "질문:" not in b or "답변:" not in b:
            continue
        header = re.match(r"\[([^\]]+)\]\s*\(([^)]+)\)", b)
        q = re.search(r"질문:\s*(.+?)(?=\n(?:Reference|답변:))", b, re.S)
        ref_n = re.search(r"Reference Documents \(Top (\d+)\):", b)
        sources = re.findall(r"ID/Source=([^\n]+)", b)
        a = re.search(r"답변:\s*(.+)$", b, re.S)
        if not (q and a):
            continue
        q_text = q.group(1).strip()
        a_text = a.group(1).strip()
        out.append(
            {
                "timestamp": header.group(1) if header else "",
                "conv_meta": header.group(2) if header else "",
                "month": path.stem.replace("chat_logs_", ""),
                "q": q_text,
                "q_len": len(q_text),
                "a_lines": a_text.count("\n") + 1,
                "a_len": len(a_text),
                "top_n": int(ref_n.group(1)) if ref_n else 0,
                "sources": sources[: min(5, len(sources))],
                "has_table": "|" in a_text and "---" in a_text,
                "has_keyword_block": "핵심 키워드" in a_text,
                "has_gcb_footer": "GCB에 등록" in a_text,
                "has_cause_section": bool(re.search(r"##+\s*(원인|Cause|현상)", a_text)),
                "has_summary": bool(re.search(r"##+\s*(요약|결론|Summary)", a_text)),
                "has_part_num": bool(re.search(r"Part\s*Number|KA\d+|부품.*번호", a_text)),
                "first_line": a_text.split("\n", 1)[0][:80],
                "answer_full": mask_pii(a_text),
            }
        )
    return out


def classify_question(q: str) -> str:
    ql = q.lower()
    if len(q) <= 10:
        return "short_followup"
    if re.search(r"part\s*number|파트.*번호|cno|mm스펙|스펙\s*알려", ql):
        return "spec_inquiry"
    if re.search(r"알람|에러|alarm|error|fail|fault|이상", ql):
        return "alarm_trouble"
    if re.search(r"이력|언제|누가|누구|작업한|작업했|작업.*history", ql):
        return "history_lookup"
    if re.search(r"어디|위치|어느|location", ql):
        return "location_inquiry"
    if re.search(r"교체|설치|제거|분해|청소|세척|교환|setup|install", ql):
        return "procedure"
    if re.search(r"원인|방법|어떻게|왜|해결", ql):
        return "troubleshoot_diag"
    if re.search(r"목록|리스트|list|뭐가 있|어떤.*있", ql):
        return "list_lookup"
    return "general"


def pick_diverse_samples(records: list[dict[str, Any]], n: int = 20) -> list[dict[str, Any]]:
    """질문 유형별로 고르게, 답변 길이도 골고루 선택."""
    # 질문 유형 부여
    for r in records:
        r["qtype"] = classify_question(r["q"])

    # 답변 길이 bucket
    def length_bucket(lines: int) -> str:
        if lines <= 5:
            return "xs"
        if lines <= 20:
            return "s"
        if lines <= 60:
            return "m"
        if lines <= 120:
            return "l"
        return "xl"

    for r in records:
        r["a_bucket"] = length_bucket(r["a_lines"])

    # 월별 / 유형 / 길이 bucket 골고루
    import random

    random.seed(42)
    random.shuffle(records)

    targets = [
        # (qtype, a_bucket, count) — 총 20
        ("spec_inquiry", "s", 2),
        ("spec_inquiry", "m", 1),
        ("alarm_trouble", "m", 2),
        ("alarm_trouble", "l", 2),
        ("history_lookup", "m", 2),
        ("location_inquiry", "s", 1),
        ("location_inquiry", "m", 1),
        ("procedure", "m", 2),
        ("procedure", "l", 1),
        ("troubleshoot_diag", "l", 2),
        ("troubleshoot_diag", "xl", 1),
        ("list_lookup", "m", 1),
        ("short_followup", "s", 1),
        ("general", "m", 1),
    ]
    chosen: list[dict[str, Any]] = []
    for qtype, bucket, cnt in targets:
        pool = [r for r in records if r["qtype"] == qtype and r["a_bucket"] == bucket]
        chosen.extend(pool[:cnt])

    # 20 미달 시 나머지 유형으로 보충
    if len(chosen) < n:
        chosen_ids = {id(r) for r in chosen}
        for r in records:
            if id(r) in chosen_ids:
                continue
            chosen.append(r)
            if len(chosen) >= n:
                break
    return chosen[:n]


def dump_samples(samples: list[dict[str, Any]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # JSON (full)
    (OUT_DIR / "samples.json").write_text(
        json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Markdown (readable)
    lines = ["# Chat Log Evaluation Samples (n=" f"{len(samples)})", ""]
    lines.append("| # | month | qtype | bucket | q_len | a_lines | has_table | has_kw_block | q 요약 |")
    lines.append("|---|---|---|---|---:|---:|:---:|:---:|---|")
    for i, r in enumerate(samples, 1):
        q_short = r["q"].replace("\n", " ")[:60]
        lines.append(
            f"| {i} | {r['month']} | {r['qtype']} | {r['a_bucket']} | {r['q_len']} | {r['a_lines']} | "
            f"{'Y' if r['has_table'] else '.'} | {'Y' if r['has_keyword_block'] else '.'} | {q_short} |"
        )
    lines.append("")
    lines.append("---")
    lines.append("")

    for i, r in enumerate(samples, 1):
        lines.append(f"## Sample {i:02d} — {r['qtype']} / {r['a_bucket']} ({r['a_lines']}줄)")
        lines.append("")
        lines.append(f"- **Timestamp**: `{r['timestamp']}`")
        lines.append(f"- **Question** ({r['q_len']}자):")
        lines.append("")
        lines.append("  ```")
        for ql in r["q"].split("\n"):
            lines.append(f"  {ql}")
        lines.append("  ```")
        lines.append(f"- **Sources (Top {r['top_n']})**: first 5 = {r['sources']}")
        lines.append(
            f"- **Structure flags**: table={'Y' if r['has_table'] else '.'} "
            f"kw_block={'Y' if r['has_keyword_block'] else '.'} "
            f"summary={'Y' if r['has_summary'] else '.'} "
            f"cause={'Y' if r['has_cause_section'] else '.'} "
            f"gcb_footer={'Y' if r['has_gcb_footer'] else '.'} "
            f"part_num={'Y' if r['has_part_num'] else '.'}"
        )
        lines.append(f"- **First line**: `{r['first_line']}`")
        lines.append("")
        lines.append("<details><summary>Reference answer (PII masked)</summary>")
        lines.append("")
        lines.append("```markdown")
        lines.extend(r["answer_full"].split("\n"))
        lines.append("```")
        lines.append("")
        lines.append("</details>")
        lines.append("")

    (OUT_DIR / "samples.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    all_records: list[dict[str, Any]] = []
    for fp in LOG_FILES:
        all_records.extend(parse_records(fp))
    print(f"총 파싱 레코드: {len(all_records)}")

    samples = pick_diverse_samples(all_records, n=20)
    print(f"선정 샘플: {len(samples)}")
    from collections import Counter

    tbl = Counter((r["qtype"], r["a_bucket"]) for r in samples)
    for (qt, b), c in sorted(tbl.items()):
        print(f"  {qt}/{b}: {c}")

    dump_samples(samples)
    print(f"저장: {OUT_DIR}")


if __name__ == "__main__":
    main()
