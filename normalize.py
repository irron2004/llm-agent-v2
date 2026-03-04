"""파일명 정규화 + manifest 생성 스크립트.

원본 데이터 파일들의 이름에서 doc_type, device_name, work_type, module, topic을
정규화된 형태로 추출하여 manifest JSON을 생성한다.

파일명 구조:
    SOP: Global SOP_{device}_{work_type}_{module}_{topic}.pdf
         work_type: ADJ(Adjust), REP(Replace), CLN(Clean), SW(Software), FA(F/A측정), ALL
         module: PM, EFEM, LL, TM, SUB, AM, CHAMBER, ALL
    TS:  {device}_{module}_Trouble_Shooting_Guide_{topic}.pdf
    Setup Manual: Set_Up_Manual_{device}.pdf

Usage:
    python scripts/chunk_v3/normalize.py \
        --data-root /home/llm-share/datasets/pe_agent_data/pe_preprocess_data \
        --output data/chunk_v3_manifest.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# =============================================================================
# 장비명 정규화 매핑
# =============================================================================

DEVICE_ALIAS_MAP: dict[str, str] = {
    "geneva": "GENEVA",
    "geneva xp": "GENEVA_XP",
    "genevaxp": "GENEVA_XP",
    "geneva stp300 xp": "GENEVA_STP300_XP",
    "integer": "INTEGER",
    "integer plus": "INTEGER_PLUS",
    "supra": "SUPRA",
    "supra n": "SUPRA_N",
    "supra nm": "SUPRA_NM",
    "supra np": "SUPRA_NP",
    "supra n series": "SUPRA_N",
    "supra series": "SUPRA",
    "supra v": "SUPRA_V",
    "supra vm": "SUPRA_VM",
    "supra vplus": "SUPRA_VPLUS",
    "supra v plus": "SUPRA_VPLUS",
    "supra xp": "SUPRA_XP",
    "supra xq": "SUPRA_XQ",
    "supran": "SUPRA_N",
    "precia": "PRECIA",
    "omnis": "OMNIS",
    "omnis plus": "OMNIS_PLUS",
    "ecolite 2000": "ECOLITE_2000",
    "ecolite 3000": "ECOLITE_3000",
    "ecolite ii 400": "ECOLITE_II_400",
    "zedius xp": "ZEDIUS_XP",
    "zedius xp(supra xp)": "ZEDIUS_XP",
    "all": "ALL",
}

WORK_TYPES = {"ADJ", "REP", "CLN", "SW", "FA", "ALL", "MODIFY", "MFG"}
MODULES = {"PM", "EFEM", "LL", "TM", "SUB", "AM", "CHAMBER", "ALL", "RACK"}

# =============================================================================
# 문서 내부 확인 기반 수동 오버라이드 (파일명 파싱 한계 보정)
# key: 파일명, value: 덮어쓸 필드 dict
# =============================================================================

MANUAL_OVERRIDES: dict[str, dict[str, str]] = {
    # --- work_type 비어있던 OMNIS_PLUS 5건 (문서 타이틀에서 확인) ---
    "global sop_omnis plus_pm aging_eng.pdf": {"work_type": "ALL"},
    "global sop_omnis plus_pm_apc valve setting_eng.pdf": {"work_type": "ADJ"},
    "global sop_omnis plus_pm_er_check_eng.pdf": {"work_type": "ALL"},
    "global sop_omnis plus_pm_gap sensor_eng.pdf": {"work_type": "ADJ"},
    "global sop_omnis plus_tm_awc sensor_r0.pdf": {"work_type": "ADJ"},
    # --- FA → REP (문서 scope: "replacement operation of Formic bottle") ---
    "Global SOP_GENEVA xp_FA measurement_EN.pdf": {"work_type": "REP", "module": "SUB"},
    # --- module 비어있던 SOP 파일들 (문서 내부 확인) ---
    # Heater Chuck / Chuck 관련 → PM
    "Global SOP_GENEVA xp_ADJ_Heater Chuck leveling.pdf": {"module": "PM"},
    "global sop_genevaxp_rep_chuck hard stopper.pdf": {"module": "PM"},
    "global sop_omnis plus_all_chuck temp_product report.pdf": {"module": "PM"},
    # 8계통 전체 점검 → ALL
    "Global SOP_GENEVA xp_ALL_8계통 Check.pdf": {"module": "ALL"},
    # Bubbler Cabinet 관련 → SUB
    "Global SOP_GENEVA xp_REP_BUBBLER CABINET_DRAIN VALVE.pdf": {"module": "SUB"},
    "Global SOP_GENEVA xp_REP_Bubbler Cabinet_Feed Valve.pdf": {"module": "SUB"},
    "Global SOP_GENEVA_REP_Bubbler Cabinet_Fill valve.pdf": {"module": "SUB"},
    "Global SOP_GENEVA_REP_Bubbler Cabinet_Safety valve.pdf": {"module": "SUB"},
    "Global SOP_GENEVA_REP_Bubbler Cabinet_Vent valve.pdf": {"module": "SUB"},
    "global sop_geneva xp_cln_bubbler cabinet_canister.pdf": {"module": "SUB"},
    "global sop_geneva xp_rep_bubbler cabinet_delivery valve.pdf": {"module": "SUB"},
    "global sop_geneva xp_rep_bubbler cabinet_formic detector cartridge.pdf": {"module": "SUB"},
    "global sop_geneva xp_rep_bubbler cabinet_formic detector.pdf": {"module": "SUB"},
    "global sop_geneva xp_rep_bubbler cabinet_relief valve.pdf": {"module": "SUB"},
    # APC valve → PM
    "global sop_geneva xp_adj_apc auto calibration.pdf": {"module": "PM"},
    # Post align → EFEM
    "global sop_geneva xp_adj_post align application.pdf": {"module": "EFEM"},
    # CTC 교체 (Working Model: GENEVA STP300 XP) → PM
    "Global_SOP_GENEVA_REP_CTC.pdf": {"device_name": "GENEVA_STP300_XP", "module": "PM"},
    # Generator 관련 → PM (work list: PM Water Shut Off Valve)
    "global sop_integer plus_all_gr_generator.pdf": {"module": "PM"},
    "global sop_integer plus_all_gr_shut off valve.pdf": {"module": "PM"},
    # Smart Match (MW 부품) → PM
    "global sop_supra series_all_smart match.pdf": {"module": "PM"},
    # SW Operation (범용) → ALL
    "global sop_supra series_all_sw operation.pdf": {"module": "ALL"},
    # RACK 관련 → RACK
    "global sop_supra n series_all_rack.pdf": {"module": "RACK"},
    "global sop_supra xp_all_rack_rf generator.pdf": {"device_name": "ZEDIUS_XP", "module": "RACK"},
    # PCW Turn On (PM + Cooling Stage) → ALL
    "global_sop_supra_n series_all_pcw_turn_on.pdf": {"module": "ALL"},
    # --- PPTX module 비어있던 파일들 ---
    "Global SOP_SUPRA Vplus_ADJ_LOAD PORT CERTIFICATION_ENG.pptx": {"module": "EFEM"},
    "Global SOP_SUPRA Vplus_MODIFY_ARM LEVELING SENSOR_EN.pptx": {"module": "EFEM"},
    "Global SOP_SUPRA Vplus_MODIFY_SU_LOTO COVER_EN.pptx": {"module": "ALL"},
    "Global SOP_SUPRA Vplus_REP_RACK ELB_ENG.pptx": {"module": "RACK"},
    "Global SOP_SUPRA Vplus_REP_RACK_DC POWER SUPPLY_ENG.pptx": {"module": "RACK"},
    "Global SOP_SUPRA Vplus_REP_RACK_MCB_ENG.pptx": {"module": "RACK"},
    "Global SOP_SUPRA Vplus_REP_RACK_SAFETY MODULE_ENG.pptx": {"module": "RACK"},
    "Global SOP_SUPRA Vplus_REP_RACK_UPS Battery_ENG.pptx": {"module": "RACK"},
    # --- module_rare 수정 ---
    # SUB_UNIT → SUB (통일)
    "SUPRA XP_SUB UNIT_Trouble_Shooting_Guide_IGS Block Abnormal.pdf": {"module": "SUB"},
}


def normalize_device_name(raw: str) -> str:
    """장비명 정규화."""
    if not raw:
        return ""
    cleaned = raw.strip()
    cleaned = re.sub(r"\(.*?\)", "", cleaned).strip()
    key_spaces = cleaned.lower().replace("_", " ").strip()
    key_spaces = re.sub(r"\s+", " ", key_spaces)

    if key_spaces in DEVICE_ALIAS_MAP:
        return DEVICE_ALIAS_MAP[key_spaces]

    best_match = ""
    best_normalized = ""
    for alias, normalized in DEVICE_ALIAS_MAP.items():
        if key_spaces.startswith(alias) and len(alias) > len(best_match):
            best_match = alias
            best_normalized = normalized
    if best_normalized:
        return best_normalized

    result = re.sub(r"[^a-zA-Z0-9가-힣]+", "_", cleaned)
    result = re.sub(r"_+", "_", result).strip("_").upper()
    return result


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


_KNOWN_DEVICES_LOWER = sorted(DEVICE_ALIAS_MAP.keys(), key=len, reverse=True)


def _split_device_and_rest(text: str) -> tuple[str, str]:
    text_lower = text.lower().replace("_", " ").strip()
    text_lower = re.sub(r"\s+", " ", text_lower)

    for device_key in _KNOWN_DEVICES_LOWER:
        if text_lower.startswith(device_key):
            rest_start = len(device_key)
            original_normalized = text.replace("_", " ").strip()
            original_normalized = re.sub(r"\s+", " ", original_normalized)
            rest = original_normalized[rest_start:].strip()
            device_raw = original_normalized[:rest_start].strip()
            rest = re.sub(r"^[\s_]+", "", rest)
            return device_raw, rest

    parts = re.split(r"[\s_]", text, maxsplit=1)
    device_raw = parts[0].strip()
    rest = parts[1].strip() if len(parts) > 1 else ""
    return device_raw, rest


def _split_work_type_module_topic(rest: str) -> tuple[str, str, str]:
    tokens = re.split(r"[\s_]+", rest, maxsplit=2)
    work_type = ""
    module = ""
    topic = rest

    if not tokens:
        return work_type, module, topic

    first_upper = tokens[0].upper()

    # "SET UP" multi-word work_type
    two_word = (tokens[0] + " " + tokens[1]).upper() if len(tokens) >= 2 else ""
    if two_word in ("SET UP", "SET_UP"):
        work_type = "SETUP"
        skip_len = len(tokens[0]) + 1 + len(tokens[1]) if len(tokens) >= 2 else len(tokens[0])
        remaining = rest[skip_len:].strip().lstrip("_ ")
        remaining_tokens = re.split(r"[\s_]+", remaining, maxsplit=1) if remaining else []
        if remaining_tokens and remaining_tokens[0].upper() in MODULES:
            module = remaining_tokens[0].upper()
            topic = remaining_tokens[1].strip() if len(remaining_tokens) > 1 else ""
        else:
            topic = remaining
        return work_type, module, _normalize_text(topic)

    # 첫 토큰이 work_type
    if first_upper in WORK_TYPES:
        work_type = first_upper
        remaining = rest[len(tokens[0]):].strip().lstrip("_ ")

        if not remaining:
            return work_type, module, ""

        remaining_tokens = re.split(r"[\s_]+", remaining, maxsplit=1)

        if remaining_tokens[0].upper() in MODULES:
            module = remaining_tokens[0].upper()
            topic = remaining_tokens[1].strip() if len(remaining_tokens) > 1 else ""
        else:
            if remaining.upper().startswith("SUB UNIT") or remaining.upper().startswith("SUB_UNIT"):
                module = "SUB_UNIT"
                topic = remaining[len("SUB UNIT"):].strip().lstrip("_ ")
            else:
                topic = remaining

    # 첫 토큰이 module (work_type 없이)
    elif first_upper in MODULES:
        module = first_upper
        remaining = rest[len(tokens[0]):].strip().lstrip("_ ")
        topic = remaining

    else:
        topic = rest

    return work_type, module, _normalize_text(topic)


def _parse_sop_filename(filename: str) -> dict[str, str]:
    stem = Path(filename).stem
    match = re.match(r"[Gg][Ll]?[Oo][Bb][Aa][Ll][\s_]+[Ss][Oo][Pp][\s_]+(.+)", stem)
    if not match:
        return {"device_name": "", "work_type": "", "module": "", "topic": ""}

    rest = match.group(1)
    device, after_device = _split_device_and_rest(rest)

    after_device = re.sub(r"[\s_]+(EN|KR|ENG)\s*$", "", after_device, flags=re.IGNORECASE)
    after_device = re.sub(r"[\s_]+R\d+\s*$", "", after_device, flags=re.IGNORECASE)
    after_device = re.sub(r"[\s_]+JP\s*$", "", after_device, flags=re.IGNORECASE)

    work_type, module, topic = _split_work_type_module_topic(after_device)

    return {
        "device_name": normalize_device_name(device),
        "work_type": work_type,
        "module": module,
        "topic": _normalize_text(topic),
    }


def _parse_ts_filename(filename: str) -> dict[str, str]:
    stem = Path(filename).stem
    match = re.match(
        r"(.+?)_(ALL|PM|LL|EFEM|TM|SUB[\s_]*UNIT)_Trouble[\s_]*Shoot(?:ing)?[\s_]*(?:Guide)?[\s_]*(.*)",
        stem, re.IGNORECASE,
    )
    if not match:
        return {"device_name": "", "work_type": "TS", "module": "", "topic": stem}

    device_raw = match.group(1).strip()
    module = match.group(2).upper().replace(" ", "_")
    topic = match.group(3).strip()
    topic = re.sub(r"^Trace[\s_]*", "", topic, flags=re.IGNORECASE)

    return {
        "device_name": normalize_device_name(device_raw),
        "work_type": "TS",
        "module": module,
        "topic": _normalize_text(topic),
    }


def _parse_setup_manual_filename(filename: str) -> dict[str, str]:
    stem = Path(filename).stem
    match = re.match(r"Set[\s_\-]+Up[\s_]+Manual[\s_]+(.*)", stem, re.IGNORECASE)
    if not match:
        return {"device_name": "", "work_type": "SETUP", "module": "", "topic": ""}

    device_raw = match.group(1).strip()
    device_raw = re.sub(r"[\s_]+R\d+\s*$", "", device_raw)

    return {
        "device_name": normalize_device_name(device_raw),
        "work_type": "SETUP",
        "module": "",
        "topic": "",
    }


def _apply_overrides(entry: dict[str, Any]) -> dict[str, Any]:
    """MANUAL_OVERRIDES에서 파일명 매칭하여 필드 덮어쓰기."""
    overrides = MANUAL_OVERRIDES.get(entry["file_name"])
    if overrides:
        for key, val in overrides.items():
            entry[key] = val
    return entry


def build_manifest(data_root: Path) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []

    sop_dir = data_root / "sop_pdfs"
    if sop_dir.exists():
        for f in sorted(sop_dir.iterdir()):
            if f.suffix.lower() not in (".pdf", ".pptx"):
                continue
            meta = _parse_sop_filename(f.name)
            doc_type = "sop_pptx" if f.suffix.lower() == ".pptx" else "sop_pdf"
            manifest.append(_apply_overrides({
                "file_path": str(f), "file_name": f.name, "doc_type": doc_type,
                "device_name": meta["device_name"], "work_type": meta["work_type"],
                "module": meta["module"], "topic": meta["topic"],
                "source_type": f.suffix.lower().lstrip("."),
            }))

    ts_dir = data_root / "ts_pdfs"
    if ts_dir.exists():
        for f in sorted(ts_dir.iterdir()):
            if f.suffix.lower() != ".pdf":
                continue
            meta = _parse_ts_filename(f.name)
            manifest.append(_apply_overrides({
                "file_path": str(f), "file_name": f.name, "doc_type": "ts",
                "device_name": meta["device_name"], "work_type": meta["work_type"],
                "module": meta["module"], "topic": meta["topic"],
                "source_type": "pdf",
            }))

    setup_dir = data_root / "set_up_manual"
    if setup_dir.exists():
        for f in sorted(setup_dir.iterdir()):
            if f.suffix.lower() != ".pdf":
                continue
            meta = _parse_setup_manual_filename(f.name)
            manifest.append(_apply_overrides({
                "file_path": str(f), "file_name": f.name, "doc_type": "setup_manual",
                "device_name": meta["device_name"], "work_type": meta["work_type"],
                "module": meta["module"], "topic": meta["topic"],
                "source_type": "pdf",
            }))

    myservice_dir = data_root / "myservice_txt"
    if myservice_dir.exists():
        for f in sorted(myservice_dir.iterdir()):
            if f.suffix.lower() != ".txt":
                continue
            manifest.append({
                "file_path": str(f), "file_name": f.name, "doc_type": "myservice",
                "device_name": "", "work_type": "", "module": "", "topic": "",
                "source_type": "txt",
            })

    gcb_dir = data_root / "gcb_raw"
    if gcb_dir.exists():
        for f in sorted(gcb_dir.rglob("*.json")):
            manifest.append({
                "file_path": str(f), "file_name": f.name, "doc_type": "gcb",
                "device_name": "", "work_type": "", "module": "", "topic": "",
                "source_type": "json",
            })

    return manifest


def print_stats(manifest: list[dict[str, Any]]) -> None:
    from collections import Counter
    doc_entries = [m for m in manifest if m["doc_type"] not in ("myservice", "gcb")]

    print(f"\nTotal files: {len(manifest)}")

    type_counts = Counter(m["doc_type"] for m in manifest)
    print("\nBy doc_type:")
    for dt, cnt in sorted(type_counts.items()):
        print(f"  {dt}: {cnt}")

    device_counts = Counter(m["device_name"] for m in doc_entries if m["device_name"])
    print(f"\nUnique device names (SOP/TS/Setup): {len(device_counts)}")
    for dev, cnt in sorted(device_counts.items()):
        print(f"  {dev}: {cnt}")

    wt_counts = Counter(m["work_type"] for m in doc_entries if m["work_type"])
    print(f"\nBy work_type:")
    for wt, cnt in sorted(wt_counts.items()):
        print(f"  {wt}: {cnt}")

    mod_counts = Counter(m["module"] for m in doc_entries if m["module"])
    print(f"\nBy module:")
    for mod, cnt in sorted(mod_counts.items()):
        print(f"  {mod}: {cnt}")

    empty_device = [m for m in doc_entries if not m["device_name"]]
    if empty_device:
        print(f"\nWARNING: {len(empty_device)} files with empty device_name:")
        for m in empty_device[:10]:
            print(f"  [{m['doc_type']}] {m['file_name']}")

    sop_no_wt = [m for m in doc_entries if m["doc_type"].startswith("sop") and not m["work_type"]]
    if sop_no_wt:
        print(f"\nWARNING: {len(sop_no_wt)} SOP files with empty work_type:")
        for m in sop_no_wt[:10]:
            print(f"  {m['file_name']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="파일명 정규화 manifest 생성")
    parser.add_argument("--data-root", default="/home/llm-share/datasets/pe_agent_data/pe_preprocess_data")
    parser.add_argument("--output", default="data/chunk_v3_manifest.json")
    parser.add_argument("--stats-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)

    print(f"Scanning: {data_root}")
    manifest = build_manifest(data_root)
    print_stats(manifest)

    if not args.stats_only:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        print(f"\nManifest saved: {output} ({len(manifest)} entries)")


if __name__ == "__main__":
    main()
