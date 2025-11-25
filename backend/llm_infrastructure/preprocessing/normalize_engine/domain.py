"""Domain-specific normalization implementations (L3/L4/L5)."""

from __future__ import annotations

import re
import unicodedata
from statistics import mean
from typing import Dict, Optional

from .base import clean_variants
from .rules import JOIN_RULES, L5_VARIANT_MAP


def prejoin_domain_terms(text: str) -> str:
    """도메인 패턴을 하나의 토큰으로 결합합니다."""
    t = text
    for pat, rep in JOIN_RULES:
        t = re.sub(pat, rep, t, flags=re.IGNORECASE)
    return t


def preprocess_semiconductor_domain(
    text: str, variant_map: Optional[Dict[str, str]] = None
) -> str:
    """
    Semiconductor maintenance log specialized preprocessing (L3).

    Includes:
    - PM module masking for embedding
    - Domain cleanup and synonyms
    - Entity extraction hints preservation
    - Scientific notation normalization
    """
    if not text:
        return ""

    # 1. Unicode/whitespace/dash normalization
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[‐‑‒–—−]", "-", text)  # Dash normalization
    text = text.replace("→", "->").replace("∼", "~")

    # 2. Domain fixes / synonyms
    domain_fixes = [
        (re.compile(r"\bexhasut\b", re.I), "exhaust"),
        (re.compile(r"\bgas box exhasut\b", re.I), "gas box exhaust"),
        (re.compile(r"\bfdc\b", re.I), "FDC"),
        (re.compile(r"\bsec\s*p3d\b", re.I), "SEC P3D"),
        (re.compile(r"\bsec-?p3\((?:d)\)\(kr\)", re.I), "SEC-P3D"),
        (re.compile(r"\bsec-?p2\((?:d)\)\(kr\)", re.I), "SEC-P2D"),
        (re.compile(r"\bsec-?p1\((?:d)\)\(kr\)", re.I), "SEC-P1D"),
        (re.compile(r"\bp\.?\s*/?\s*c\b", re.I), "pc"),
        (re.compile(r"\bv\s?tis", re.I), "vtis"),
    ]
    for pat, repl in domain_fixes:
        text = pat.sub(repl, text)

    # 3. PM module masking (pm2, pm2-1 -> PM for embedding)
    pm_addr_re = re.compile(r"\bpm\s*(\d+)(?:\s*[-_.\/]\s*(\d+))?\b", re.IGNORECASE)
    text = pm_addr_re.sub("PM", text)

    # 4. Scientific notation normalization
    sci_not_re = re.compile(r"(\d+(?:\.\d+)?)\s*[×x]\s*10\^?\s*([-+]?\d+)")

    def normalize_sci_notation(m):
        base = float(m.group(1))
        exp = int(m.group(2))
        return str(base * (10**exp))

    text = sci_not_re.sub(normalize_sci_notation, text)

    # 5. Apply variant mapping if provided
    if variant_map:
        text = clean_variants(text, variant_map)

    # 6. Final whitespace normalization
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_l4_advanced_domain(
    text: str, variant_map: Optional[Dict[str, str]] = None
) -> str:
    """
    Advanced semiconductor domain preprocessing (L4) with enhanced entity extraction.
    """
    if not text:
        return ""

    s = unicodedata.normalize("NFKC", text)
    s = re.sub(r"[‐‑‒–—−]", "-", s)  # Dash normalization
    s = s.replace("→", "->").replace("∼", "~").replace("‐", "-")
    s = re.sub(r"\s+", " ", s).strip()

    # Expanded domain fixes (enhanced synonym dictionary)
    domain_fixes = {
        # typos
        "exhasut": "exhaust",
        "opne": "open",
        "clos e": "close",
        # synonyms/variants
        "slot v/v": "slot valve",
        "slot vv": "slot valve",
        "vv": "valve",
        "apc": "APC",
        "fdc": "FDC",
        "he leak": "helium leak",
        "b.g": "baratron",
        "b g": "baratron",
        "bara": "baratron",
        "pirani gague": "pirani gauge",
        "pc 현상": "particle",
        "p.c 현상": "particle",
        "p.c": "particle",
        "p/ c": "particle",
        "spec in": "SPECIN",
        "spec out": "SPECOUT",
    }
    for a, b in domain_fixes.items():
        s = re.sub(rf"\b{re.escape(a)}\b", b, s, flags=re.IGNORECASE)

    # Module/chamber pattern extraction and normalization
    module_re = re.compile(
        r"\b((?:pm|am|tm|ll|lp)\s*-?\s*\d(?:-\d)?|(?:pm|am)\s*ch\s*\d|efem)\b",
        re.IGNORECASE,
    )
    modules = []
    for m in module_re.findall(s):
        m0 = re.sub(r"\s+", "", m.upper())
        m0 = m0.replace("CH", "-CH") if "CH" in m0 and "-CH" not in m0 else m0
        modules.append(m0)

    # Alarm/interlock extraction
    alarm_name_re = re.compile(
        r"\b([a-z0-9 /_-]*?(?:alarm|interlock|timeout|abort|fail(?:ed)?))\b",
        re.IGNORECASE,
    )
    alarm_code_re = re.compile(r"\((\d{3,7})\)")
    alarms = [a.lower().replace(" ", "_") for a in alarm_name_re.findall(s)]
    alarm_codes = alarm_code_re.findall(s)

    # Helium leak
    sci_mul = r"[×x\*]"
    he_leak_re = re.compile(
        rf"helium leak(?: check)?\s*[:=]?\s*"
        rf"(?P<base>\d+(?:\.\d+)?)\s*(?:{sci_mul}\s*10\^[-−]?\s*(?P<exp>\d+))?"
        rf"(?:\s*mbar\*?l/s)?",
        re.IGNORECASE,
    )
    he_leak_val = None
    for m in he_leak_re.finditer(s):
        base = float(m.group("base"))
        exp = m.group("exp")
        he_leak_val = base if exp is None else base * (10 ** (-int(exp)))
        break

    # Pressure in mTorr
    press_mt_re = re.compile(r"(?<!\w)(\d+(?:\.\d+)?)\s*(?:m?torr|mt)\b", re.IGNORECASE)
    press_values = [float(x) for x in press_mt_re.findall(s)]

    # Open/close time (sec)
    oc_time_re = re.compile(
        r"(?:open\s*/\s*close|open/close|open|close)\s*time\s*[:=]?\s*(\d+(?:\.\d+)?)\s*s",
        re.IGNORECASE,
    )
    oc_times = [float(x) for x in oc_time_re.findall(s)]
    _ = oc_times  # reserved for future features

    # Speed change
    speed_change_re = re.compile(
        r"\bspeed[^:]*[:=]?\s*(\d+(?:\.\d+)?)\s*[-–>]+\s*(\d+(?:\.\d+)?)",
        re.IGNORECASE,
    )
    speed_changes = [(float(a), float(b)) for a, b in speed_change_re.findall(s)]
    _ = speed_changes

    # Range/series processing
    range_re = re.compile(r"(-?\d+(?:\.\d+)?)\s*[-~]\s*(-?\d+(?:\.\d+)?)")
    ranges = []
    for a, b in range_re.findall(s):
        try:
            ranges.append((float(a), float(b)))
        except Exception:
            pass

    series_re = re.compile(r"(-?\d+(?:\.\d+)?)(?:\s*/\s*(-?\d+(?:\.\d+)?))+")
    all_values = []
    for m in series_re.finditer(s):
        full_match = m.group(0)
        seq = re.split(r"\s*/\s*", full_match)
        try:
            vals = [float(v) for v in seq if v.strip()]
            all_values.extend(vals)
        except Exception:
            pass

    # Spec in/out detection
    spec_in_re = re.compile(r"\bSPECIN\b|\b이상 ?없(?:음|는)|\bpass\b", re.IGNORECASE)
    spec_out_re = re.compile(r"\bSPECOUT\b|\b불량|spec out|out of spec", re.IGNORECASE)
    spec_status = None
    if spec_in_re.search(s):
        spec_status = "IN"
    elif spec_out_re.search(s):
        spec_status = "OUT"

    # Action type labeling
    action_keywords = {
        "REP": ["rep", "교체"],
        "ADJ": ["adj", "조정", "adjust"],
        "CLN": ["cln", "clean", "세정"],
        "CAL": ["cal", "켈", "calibration", "teaching"],
        "PATCH": ["patch", "패치"],
        "CHK": ["check", "점검"],
        "MON": ["monitor", "모니터링"],
        "BKUP": ["backup", "백업"],
    }
    s_low = s.lower()
    actions = {tag for tag, kws in action_keywords.items() if any(k in s_low for k in kws)}

    # Build structured tokens for embedding header
    header_tokens = []
    if modules:
        header_tokens.extend([f"[MODULE {m}]" for m in sorted(set(modules))])
    if alarm_codes:
        header_tokens.extend([f"[ALARM {c}]" for c in sorted(set(alarm_codes))])
    if alarms:
        header_tokens.extend([f"[ALARM_NAME {a}]" for a in sorted(set(alarms))])
    if spec_status:
        header_tokens.append(f"[SPEC {spec_status}]")
    if he_leak_val is not None:
        header_tokens.append(f"[HE_LEAK {he_leak_val:.2e}]")
    if press_values:
        header_tokens.append(f"[PRESS_MT_MAX {max(press_values):.0f}]")
    if all_values:
        try:
            header_tokens.append(f"[VAL_MIN {min(all_values)}]")
            header_tokens.append(f"[VAL_MAX {max(all_values)}]")
            header_tokens.append(f"[VAL_AVG {round(mean(all_values), 3)}]")
        except Exception:
            pass
    for act in sorted(actions):
        header_tokens.append(f"[ACTION {act}]")

    # Apply variant mapping if provided
    if variant_map:
        s = clean_variants(s, variant_map)

    # Combine header tokens with original text
    result = " ".join(header_tokens) + " :: " + s if header_tokens else s
    return re.sub(r"\s+", " ", result).strip()


def preprocess_l5_enhanced_domain(
    text: str, variant_map: Optional[Dict[str, str]] = None
) -> str:
    """Enhanced semiconductor domain preprocessing (L5) - L4 기반 + 상세한 동의어 처리."""
    if not text:
        return ""

    s = unicodedata.normalize("NFKC", text)
    s = re.sub(r"[‐‑‒–—−]", "-", s)
    s = s.replace("→", "->").replace("∼", "~").replace("‐", "-")
    s = re.sub(r"\s+", " ", s).strip()

    # L5 Enhanced variant mapping (현장 표기 동의어 사전)
    for variant, standard in L5_VARIANT_MAP.items():
        s = re.sub(rf"\b{re.escape(variant)}\b", standard, s, flags=re.IGNORECASE)

    # 단위/숫자 표기 표준화
    s = re.sub(r"(\d+)\s*(mtorr|mt)\b", r"\1 mTorr", s, flags=re.IGNORECASE)
    s = re.sub(r"(\d+)\s*(sccm)\b", r"\1 sccm", s, flags=re.IGNORECASE)
    s = re.sub(r"(\d+)\s*um\b", r"\1 µm", s, flags=re.IGNORECASE)
    s = re.sub(
        r"(\d+)\s*(s|hr|deg|w)\b",
        lambda m: f"{m.group(1)} {m.group(2).upper()}",
        s,
        flags=re.IGNORECASE,
    )

    # 범위 표기 정규화
    range_re = re.compile(r"(-?\d+(?:\.\d+)?)\s*[-~]\s*(-?\d+(?:\.\d+)?)")

    def normalize_range(match: re.Match) -> str:
        a, b = float(match.group(1)), float(match.group(2))
        lo, hi = (a, b) if a <= b else (b, a)
        return f"[RANGE {lo}..{hi}]"

    s = range_re.sub(normalize_range, s)

    # Module/chamber pattern extraction and normalization (L4와 동일)
    module_re = re.compile(
        r"\b((?:pm|am|tm|ll|lp)\s*-?\s*\d(?:-\d)?|(?:pm|am)\s*ch\s*\d|efem)\b",
        re.IGNORECASE,
    )
    modules = []
    for m in module_re.findall(s):
        m0 = re.sub(r"\s+", "", m.upper())
        m0 = m0.replace("CH", "-CH") if "CH" in m0 and "-CH" not in m0 else m0
        modules.append(m0)

    # Alarm/interlock extraction (L4와 동일)
    alarm_name_re = re.compile(
        r"\b([a-z0-9 /_-]*?(?:alarm|interlock|timeout|abort|fail(?:ed)?))\b",
        re.IGNORECASE,
    )
    alarm_code_re = re.compile(r"\((\d{3,7})\)")
    alarms = [a.lower().replace(" ", "_") for a in alarm_name_re.findall(s)]
    alarm_codes = alarm_code_re.findall(s)

    # Numerical value standardization (L4와 동일)
    sci_mul = r"[×x\*]"
    he_leak_re = re.compile(
        rf"helium leak(?: check)?\s*[:=]?\s*"
        rf"(?P<base>\d+(?:\.\d+)?)\s*(?:{sci_mul}\s*10\^[-−]?\s*(?P<exp>\d+))?"
        rf"(?:\s*mbar\*?l/s)?",
        re.IGNORECASE,
    )
    he_leak_val = None
    for m in he_leak_re.finditer(s):
        base = float(m.group("base"))
        exp = m.group("exp")
        he_leak_val = base if exp is None else base * (10 ** (-int(exp)))
        break

    press_mt_re = re.compile(r"(?<!\w)(\d+(?:\.\d+)?)\s*(?:m?torr|mt)\b", re.IGNORECASE)
    press_values = [float(x) for x in press_mt_re.findall(s)]

    oc_time_re = re.compile(
        r"(?:open\s*/\s*close|open/close|open|close)\s*time\s*[:=]?\s*(\d+(?:\.\d+)?)\s*s",
        re.IGNORECASE,
    )
    oc_times = [float(x) for x in oc_time_re.findall(s)]
    _ = oc_times

    speed_change_re = re.compile(
        r"\bspeed[^:]*[:=]?\s*(\d+(?:\.\d+)?)\s*[-–>]+\s*(\d+(?:\.\d+)?)",
        re.IGNORECASE,
    )
    speed_changes = [(float(a), float(b)) for a, b in speed_change_re.findall(s)]
    _ = speed_changes

    series_re = re.compile(r"(-?\d+(?:\.\d+)?)(?:\s*/\s*(-?\d+(?:\.\d+)?))+")
    all_values = []
    for m in series_re.finditer(s):
        full_match = m.group(0)
        seq = re.split(r"\s*/\s*", full_match)
        try:
            vals = [float(v) for v in seq if v.strip()]
            all_values.extend(vals)
        except Exception:
            pass

    spec_in_re = re.compile(r"\bSPEC IN\b|\b이상 ?없(?:음|는)|\bpass\b", re.IGNORECASE)
    spec_out_re = re.compile(r"\bSPEC OUT\b|\b불량|spec out|out of spec", re.IGNORECASE)
    spec_status = None
    if spec_in_re.search(s):
        spec_status = "IN"
    elif spec_out_re.search(s):
        spec_status = "OUT"

    action_keywords = {
        "REP": ["rep", "교체"],
        "ADJ": ["adj", "조정", "adjust"],
        "CLN": ["cln", "clean", "세정"],
        "CAL": ["cal", "켈", "calibration", "teaching"],
        "PATCH": ["patch", "패치"],
        "CHK": ["check", "점검"],
        "MON": ["monitor", "모니터링"],
        "BKUP": ["backup", "백업"],
    }
    s_low = s.lower()
    actions = {tag for tag, kws in action_keywords.items() if any(k in s_low for k in kws)}

    header_tokens = []
    if modules:
        header_tokens.extend([f"[MODULE {m}]" for m in sorted(set(modules))])
    if alarm_codes:
        header_tokens.extend([f"[ALARM {c}]" for c in sorted(set(alarm_codes))])
    if alarms:
        header_tokens.extend([f"[ALARM_NAME {a}]" for a in sorted(set(alarms))])
    if spec_status:
        header_tokens.append(f"[SPEC {spec_status}]")
    if he_leak_val is not None:
        header_tokens.append(f"[HE_LEAK {he_leak_val:.2e}]")
    if press_values:
        header_tokens.append(f"[PRESS_MT_MAX {max(press_values):.0f}]")
    if all_values:
        try:
            header_tokens.append(f"[VAL_MIN {min(all_values)}]")
            header_tokens.append(f"[VAL_MAX {max(all_values)}]")
            header_tokens.append(f"[VAL_AVG {round(mean(all_values), 3)}]")
        except Exception:
            pass
    for act in sorted(actions):
        header_tokens.append(f"[ACTION {act}]")

    if variant_map:
        s = clean_variants(s, variant_map)

    result = " ".join(header_tokens) + " :: " + s if header_tokens else s
    return re.sub(r"\s+", " ", result).strip()


__all__ = [
    "prejoin_domain_terms",
    "preprocess_semiconductor_domain",
    "preprocess_l4_advanced_domain",
    "preprocess_l5_enhanced_domain",
]
