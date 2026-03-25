"""도메인 사전 기반 약어 확장 모듈.

semicon_word.json에서 약어 → 풀네임 매핑을 런타임에 빌드하여,
쿼리 내 약어를 "풀네임 (약어)" 형태로 치환한다.

Usage:
    expander = AbbreviationExpander.from_raw_dict("data/semicon_word.json")
    result = expander.expand_query("AR 측정이 낮아요")
    # → "Aspect Ratio (AR) 측정이 낮아요"
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 너무 짧거나 일반적인 토큰은 약어 치환에서 제외
_MIN_ABBR_LENGTH = 2
_SKIP_TOKENS = frozenset(
    {
        # 일반 영어 단어와 충돌 가능
        "AI",
        "IC",
        "NA",
        "MO",
        "PR",
        "LED",
        "DIE",
    }
)


@dataclass
class AbbreviationMatch:
    """쿼리에서 감지된 약어 매칭 정보."""

    token: str  # 원본 쿼리의 토큰
    abbr_key: str  # 인덱스 키 (대문자)
    concept_id: int
    primary_eng: str  # 풀네임 (영어)
    primary_kr: str  # 풀네임 (한국어)
    ambiguous: bool  # 모호 여부


@dataclass
class ExpandResult:
    """약어 확장 결과."""

    original_query: str
    expanded_query: str
    matches: list[AbbreviationMatch] = field(default_factory=list)
    auto_expanded: list[str] = field(default_factory=list)  # 자동 치환된 약어
    ambiguous: list[str] = field(default_factory=list)  # 모호한 약어 (미래용)


# ---------------------------------------------------------------------------
# 내부 빌드 로직: semicon_word.json → concepts + abbreviation_index
# ---------------------------------------------------------------------------


def _parse_entries(data: dict[str, Any]) -> list[dict[str, Any]]:
    """semicon_word.json 파싱."""
    entries = []
    for key, v in data.items():
        abbr_raw = v.get("줄임말", "")
        eng = v.get("영어", "").strip()
        kr = v.get("한국어", "").strip()
        synonyms_raw = v.get("동의어", "")
        meaning = v.get("뜻", "").strip().split(":contentReference")[0].strip()

        abbrs: list[str] = []
        if isinstance(abbr_raw, list):
            abbrs = [a.strip() for a in abbr_raw if a.strip() and a.strip() != "-"]
        elif isinstance(abbr_raw, str) and abbr_raw.strip() and abbr_raw.strip() != "-":
            abbrs = [a.strip() for a in abbr_raw.split(",") if a.strip()]

        syns: list[str] = []
        if isinstance(synonyms_raw, list):
            syns = [s.strip() for s in synonyms_raw if s.strip()]
        elif isinstance(synonyms_raw, str) and synonyms_raw.strip():
            syns = [s.strip() for s in synonyms_raw.split(",") if s.strip()]

        entries.append(
            {
                "key": key,
                "eng": eng,
                "kr": kr,
                "abbrs": abbrs,
                "synonyms": syns,
                "meaning": meaning,
            }
        )
    return entries


def _build_index(
    entries: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    """Union-Find로 동일 개념을 병합하고, 약어 인덱스를 구축."""

    # -- Union-Find --
    parent = list(range(len(entries)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # 약어 기반 병합 (이름이 유사한 경우만 — 다른 개념 병합 방지)
    def _has_korean(s: str) -> bool:
        """한글 문자가 포함되어 있는지."""
        return bool(re.search(r"[가-힣]", s))

    def _kr_norm(s: str) -> str:
        """한국어명 정규화 (비교용)."""
        s = re.sub(r"\(.*?\)", "", s).strip()
        return re.sub(r"[^가-힣a-z0-9]", "", s.lower())

    def _eng_words(s: str) -> set[str]:
        """영어명에서 핵심 단어 추출 (소문자, 3글자 이상)."""
        s = re.sub(r"\(.*?\)", "", s).strip().lower()
        return {w for w in re.findall(r"[a-z]+", s) if len(w) >= 3}

    def _names_compatible(i: int, j: int) -> bool:
        """두 엔트리가 같은 개념인지 이름 유사성으로 판단."""
        kr_i_raw, kr_j_raw = entries[i]["kr"], entries[j]["kr"]
        kr_i = _kr_norm(kr_i_raw) if _has_korean(kr_i_raw) else ""
        kr_j = _kr_norm(kr_j_raw) if _has_korean(kr_j_raw) else ""

        # 한국어명이 둘 다 있고 포함 관계이면 같은 개념
        if kr_i and kr_j:
            if kr_i == kr_j or kr_i in kr_j or kr_j in kr_i:
                return True

        # 영어명 비교
        eng_i = entries[i]["eng"]
        eng_j = entries[j]["eng"]

        eng_i_clean = re.sub(r"\s*\(.*?\)", "", eng_i).strip()
        eng_j_clean = re.sub(r"\s*\(.*?\)", "", eng_j).strip()
        if eng_i_clean and eng_j_clean:
            # 한쪽이 약어 형태 (5자 이하, 공백 없음)이면 같은 개념
            i_is_abbr = len(eng_i_clean) <= 5 and " " not in eng_i_clean
            j_is_abbr = len(eng_j_clean) <= 5 and " " not in eng_j_clean
            if i_is_abbr or j_is_abbr:
                return True
            # 한쪽이 설명문 (60자 초과)이면 용어명이 아닌 정의 → 같은 개념
            if len(eng_i_clean) > 60 or len(eng_j_clean) > 60:
                return True

        # 핵심 단어 겹침 비교
        words_i = _eng_words(eng_i)
        words_j = _eng_words(eng_j)
        if words_i and words_j:
            overlap = words_i & words_j
            smaller = min(len(words_i), len(words_j))
            if smaller > 0 and len(overlap) >= max(1, smaller * 0.5):
                return True

        return False

    abbr_map: dict[str, list[int]] = defaultdict(list)
    for i, e in enumerate(entries):
        for a in e["abbrs"]:
            abbr_map[a.upper()].append(i)
        eng_clean = re.sub(r"\s*\(.*?\)", "", e["eng"]).strip().upper()
        if eng_clean and len(eng_clean) <= 6 and " " not in eng_clean:
            abbr_map[eng_clean].append(i)

    for indices in abbr_map.values():
        for j in range(1, len(indices)):
            if _names_compatible(indices[0], indices[j]):
                union(indices[0], indices[j])

    # 한국어명 기반 병합
    def kr_core(s: str) -> str:
        s = re.sub(r"\(.*?\)", "", s).strip()
        return re.sub(r"[^가-힣a-z0-9]", "", s.lower())

    kr_map: dict[str, list[int]] = defaultdict(list)
    for i, e in enumerate(entries):
        core = kr_core(e["kr"])
        if core:
            kr_map[core].append(i)
    for indices in kr_map.values():
        for j in range(1, len(indices)):
            union(indices[0], indices[j])

    # 영어명 기반 병합
    def eng_core(s: str) -> str:
        s = re.sub(r"\s*\(.*?\)", "", s).strip().upper()
        return re.sub(r"[^A-Z0-9]", "", s)

    eng_map: dict[str, list[int]] = defaultdict(list)
    for i, e in enumerate(entries):
        core = eng_core(e["eng"])
        if core and len(core) > 3:
            eng_map[core].append(i)
    for indices in eng_map.values():
        for j in range(1, len(indices)):
            union(indices[0], indices[j])

    # -- 그룹 → 개념 병합 --
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(entries)):
        groups[find(i)].append(i)

    def pick_primary_eng(all_eng: set[str]) -> str:
        clean = [e for e in all_eng if "(" not in e]
        if not clean:
            clean = list(all_eng)
        good = [e for e in clean if 3 <= len(e) <= 60]
        if good:
            named = [e for e in good if 1 <= len(e.split()) <= 6]
            if named:
                return max(named, key=len)
            return max(good, key=len)
        return min(clean, key=len) if clean else ""

    concepts: list[dict[str, Any]] = []
    for _root, member_indices in groups.items():
        members = [entries[i] for i in member_indices]
        members.sort(key=lambda x: len(x["meaning"]), reverse=True)

        all_abbrs: set[str] = set()
        all_syns: set[str] = set()
        all_eng: set[str] = set()
        all_kr: set[str] = set()

        for m in members:
            all_abbrs.update(m["abbrs"])
            all_syns.update(m["synonyms"])
            if m["eng"]:
                all_eng.add(m["eng"])
            if m["kr"]:
                all_kr.add(m["kr"])

        primary_eng = pick_primary_eng(all_eng) if all_eng else members[0]["key"]
        primary_kr = max(all_kr, key=len) if all_kr else ""
        all_abbrs = {a for a in all_abbrs if a.upper() != primary_eng.upper()}

        concepts.append(
            {
                "id": len(concepts),
                "primary_eng": primary_eng,
                "primary_kr": primary_kr,
                "abbreviations": sorted(all_abbrs, key=str.upper),
                "synonyms": sorted(all_syns),
                "meaning": members[0]["meaning"],
            }
        )

    # -- 약어 인덱스 (영어 약어 토큰) --
    abbr_index_sets: dict[str, set[int]] = defaultdict(set)
    for c in concepts:
        tokens: set[str] = set()
        for a in c["abbreviations"]:
            tokens.add(a.upper())
        for cid_check in [c]:
            for eng in [c["primary_eng"]]:
                eng_clean = re.sub(r"\s*\(.*?\)", "", eng).strip()
                if len(eng_clean) <= 5 and " " not in eng_clean:
                    tokens.add(eng_clean.upper())
        for t in tokens:
            abbr_index_sets[t].add(c["id"])

    abbr_index: dict[str, dict[str, Any]] = {}
    for token in sorted(abbr_index_sets.keys()):
        cids = sorted(abbr_index_sets[token])
        abbr_index[token] = {"concept_ids": cids, "ambiguous": len(cids) > 1}

    # -- 동의어 인덱스 (한국어/영어 동의어 → 개념 매핑) --
    synonym_index: dict[str, dict[str, Any]] = {}
    for c in concepts:
        syn_tokens: set[str] = set()
        # 동의어 추가 (한국어 1글자도 허용, 영어는 2글자 이상)
        for s in c.get("synonyms", []):
            s_clean = s.strip()
            if not s_clean:
                continue
            has_kr = bool(re.search(r"[가-힣]", s_clean))
            if has_kr or len(s_clean) >= 2:
                syn_tokens.add(s_clean)
        # 한국어명도 동의어로 취급 (primary_kr 제외한 변형들)
        # primary_kr은 치환 결과에 이미 포함되므로 제외
        for syn in syn_tokens:
            key = syn.lower()
            # 이미 약어 인덱스에 있는 영어 토큰은 건너뜀
            if syn.upper() in abbr_index:
                continue
            if key not in synonym_index:
                synonym_index[key] = {"concept_ids": [], "ambiguous": False}
            if c["id"] not in synonym_index[key]["concept_ids"]:
                synonym_index[key]["concept_ids"].append(c["id"])
                if len(synonym_index[key]["concept_ids"]) > 1:
                    synonym_index[key]["ambiguous"] = True

    return concepts, abbr_index, synonym_index


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class AbbreviationExpander:
    """도메인 사전 기반 약어 확장기."""

    def __init__(
        self,
        concepts: list[dict[str, Any]],
        abbreviation_index: dict[str, dict[str, Any]],
        synonym_index: dict[str, dict[str, Any]] | None = None,
        source_path: str = "",
        skip_tokens: frozenset[str] | None = None,
    ) -> None:
        self._concepts = concepts
        self._abbr_index = abbreviation_index
        self._synonym_index = synonym_index or {}
        self._skip_tokens = skip_tokens or _SKIP_TOKENS

        one_to_one = sum(1 for v in abbreviation_index.values() if not v.get("ambiguous"))
        one_to_n = sum(1 for v in abbreviation_index.values() if v.get("ambiguous"))

        logger.info(
            "[AbbreviationExpander] loaded from '%s': "
            "%d concepts, %d abbreviations (1:1=%d, 1:N=%d), %d synonyms",
            source_path,
            len(concepts),
            len(abbreviation_index),
            one_to_one,
            one_to_n,
            len(self._synonym_index),
        )
        if one_to_n > 0:
            ambiguous_tokens = [t for t, v in abbreviation_index.items() if v.get("ambiguous")]
            logger.info(
                "[AbbreviationExpander] ambiguous abbreviations (%d): %s",
                one_to_n,
                ", ".join(ambiguous_tokens[:20]),
            )

    @classmethod
    def from_raw_dict(cls, path: str | Path) -> AbbreviationExpander:
        """semicon_word.json에서 직접 빌드 (런타임 인덱스 생성)."""
        path = Path(path)
        if not path.exists():
            logger.warning("[AbbreviationExpander] dictionary not found: %s", path)
            return cls(concepts=[], abbreviation_index={}, source_path=str(path))

        with open(path, encoding="utf-8") as f:
            raw_data = json.load(f)

        entries = _parse_entries(raw_data)
        concepts, abbr_index, synonym_index = _build_index(entries)

        logger.info(
            "[AbbreviationExpander] built index from '%s': "
            "%d entries → %d concepts, %d abbreviation tokens, %d synonym tokens",
            path,
            len(entries),
            len(concepts),
            len(abbr_index),
            len(synonym_index),
        )

        return cls(
            concepts=concepts,
            abbreviation_index=abbr_index,
            synonym_index=synonym_index,
            source_path=str(path),
        )

    @classmethod
    def from_index(cls, path: str | Path) -> AbbreviationExpander:
        """사전 빌드된 인덱스 파일에서 로드 (하위 호환)."""
        path = Path(path)
        if not path.exists():
            logger.warning("[AbbreviationExpander] index not found: %s", path)
            return cls(concepts=[], abbreviation_index={}, source_path=str(path))

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        return cls(
            concepts=data.get("concepts", []),
            abbreviation_index=data.get("abbreviation_index", {}),
            source_path=str(path),
        )

    def expand_query(self, query: str) -> ExpandResult:
        """쿼리 내 약어/동의어를 풀네임으로 확장.

        1:1 약어는 "풀네임 (약어)" 형태로 자동 치환.
        1:N 모호 약어는 치환하지 않고 ambiguous 리스트에 기록.
        한국어/영어 동의어도 매칭하여 동일하게 처리.
        """
        if (not self._abbr_index and not self._synonym_index) or not query:
            return ExpandResult(original_query=query, expanded_query=query)

        matches: list[AbbreviationMatch] = []

        # 1) 영어 약어 토큰 매칭
        tokens = re.findall(r"\b[A-Za-z][A-Za-z0-9/·\-]*\b", query)
        seen_keys: set[str] = set()
        for token in tokens:
            key = token.upper()
            if key in seen_keys:
                continue
            if len(key) < _MIN_ABBR_LENGTH:
                continue
            if key in self._skip_tokens:
                continue

            entry = self._abbr_index.get(key)
            if not entry:
                continue

            concept_ids = entry.get("concept_ids", [])
            ambiguous = entry.get("ambiguous", False)

            if not concept_ids:
                continue

            seen_keys.add(key)

            for cid in concept_ids:
                if cid < len(self._concepts):
                    concept = self._concepts[cid]
                    matches.append(
                        AbbreviationMatch(
                            token=token,
                            abbr_key=key,
                            concept_id=cid,
                            primary_eng=concept.get("primary_eng", ""),
                            primary_kr=concept.get("primary_kr", ""),
                            ambiguous=ambiguous,
                        )
                    )

        # 2) 동의어 매칭 (한국어/영어 동의어)
        #    긴 동의어 우선 매칭 → 같은 개념에 대해 가장 긴 매칭만 유지
        if self._synonym_index:
            # 길이 내림차순 정렬하여 "캘리브레이션" > "캘리" > "캘" 순으로 매칭
            sorted_syn_keys = sorted(
                self._synonym_index.keys(),
                key=len,
                reverse=True,
            )
            matched_concept_ids: set[int] = {m.concept_id for m in matches}
            for syn_key in sorted_syn_keys:
                if syn_key not in query.lower():
                    continue
                syn_entry = self._synonym_index[syn_key]
                syn_cids = syn_entry.get("concept_ids", [])
                ambiguous = syn_entry.get("ambiguous", False)
                for cid in syn_cids:
                    if cid in matched_concept_ids:
                        continue
                    if cid < len(self._concepts):
                        concept = self._concepts[cid]
                        matches.append(
                            AbbreviationMatch(
                                token=syn_key,
                                abbr_key=f"SYN:{syn_key}",
                                concept_id=cid,
                                primary_eng=concept.get("primary_eng", ""),
                                primary_kr=concept.get("primary_kr", ""),
                                ambiguous=ambiguous,
                            )
                        )
                        matched_concept_ids.add(cid)

        if not matches:
            return ExpandResult(original_query=query, expanded_query=query)

        # 치환 적용
        expanded = query
        auto_expanded: list[str] = []
        ambiguous_list: list[str] = []

        sorted_matches = sorted(matches, key=lambda m: len(m.token), reverse=True)

        for match in sorted_matches:
            if match.ambiguous:
                ambiguous_list.append(match.token)
                logger.info(
                    "[AbbreviationExpander] AMBIGUOUS skip: '%s' in query '%s' "
                    "— candidates: concept_id=%d (%s / %s)",
                    match.token,
                    query,
                    match.concept_id,
                    match.primary_eng,
                    match.primary_kr,
                )
                continue

            if match.primary_eng.lower() in expanded.lower():
                continue

            is_synonym = match.abbr_key.startswith("SYN:")
            if is_synonym:
                # 동의어: 한국어 등 \b가 안 먹으므로 단순 문자열 치환
                idx = expanded.lower().find(match.token.lower())
                if idx == -1:
                    continue
                original_token = expanded[idx : idx + len(match.token)]
                replacement = f"{original_token} ({match.primary_eng})"
                new_expanded = expanded[:idx] + replacement + expanded[idx + len(match.token) :]
            else:
                # 영어 약어: word boundary 매칭
                pattern = re.compile(
                    rf"\b{re.escape(match.token)}\b",
                    re.IGNORECASE,
                )
                replacement = f"{match.token} ({match.primary_eng})"
                new_expanded = pattern.sub(replacement, expanded, count=1)

            if new_expanded != expanded:
                expanded = new_expanded
                auto_expanded.append(match.token)
                logger.info(
                    "[AbbreviationExpander] EXPANDED: '%s' → '%s' (%s) | query: '%s' → '%s'",
                    match.token,
                    match.primary_eng,
                    match.primary_kr,
                    query,
                    expanded,
                )

        if not auto_expanded and not ambiguous_list:
            logger.debug(
                "[AbbreviationExpander] no expansion applied for query: '%s'",
                query,
            )

        return ExpandResult(
            original_query=query,
            expanded_query=expanded,
            matches=matches,
            auto_expanded=auto_expanded,
            ambiguous=ambiguous_list,
        )

    def get_synonym_variants(
        self,
        query: str,
        *,
        max_variants: int = 2,
        abbr_selections: dict[str, str] | None = None,
    ) -> list[str]:
        """쿼리 내 약어/동의어 토큰을 대체 형태로 치환한 변형 쿼리 목록 반환.

        st_mq_node에서 search_queries에 추가하여 BM25 정확 매칭을 보장한다.
        cross-product 없이, 토큰별 가장 유용한 1개 변형만 생성.
        최대 max_variants개까지만 반환.

        예: "apc value 캘" → ["apc value calibration", "apc value cal"]
        """
        if not query or (not self._abbr_index and not self._synonym_index):
            return []

        result = self.expand_query(query)
        if not result.matches:
            return []

        variants: list[str] = []
        seen: set[str] = set()
        query_lower = query.lower()

        for match in result.matches:
            if len(variants) >= max_variants:
                break

            # 모호한 약어: abbr_selections가 있으면 해당 선택에 맞는 개념만 사용
            if match.ambiguous:
                if not abbr_selections:
                    continue
                selected = abbr_selections.get(match.abbr_key)
                if not selected:
                    continue
                # 선택된 풀네임으로 치환
                if selected.lower() not in query_lower:
                    variant = self._replace_token(query, match.token, selected)
                    if variant and variant.lower() not in seen and variant.lower() != query_lower:
                        variants.append(variant)
                        seen.add(variant.lower())
                continue

            # 1:1 매칭: concept에서 대체 형태 찾기
            cid = match.concept_id
            if cid >= len(self._concepts):
                continue
            concept = self._concepts[cid]

            token_lower = match.token.lower()
            # 대체 후보: 영어명, 한국어명, 약어, 동의어
            alternatives: list[str] = []

            eng = concept.get("primary_eng", "")
            kr = concept.get("primary_kr", "")
            abbrs = concept.get("abbreviations", [])
            syns = concept.get("synonyms", [])

            # 현재 토큰이 한국어면 → 영어 변형 우선
            is_korean_token = any('\uac00' <= c <= '\ud7a3' for c in match.token)
            if is_korean_token:
                if eng and eng.lower() != token_lower:
                    alternatives.append(eng)
                for a in abbrs:
                    if a.lower() != token_lower:
                        alternatives.append(a)
            else:
                # 영어 약어/이름 → 다른 영어 형태 또는 한국어 변형
                if eng and eng.lower() != token_lower:
                    alternatives.append(eng)
                for a in abbrs:
                    if a.lower() != token_lower:
                        alternatives.append(a)
                if kr and kr.lower() != token_lower:
                    alternatives.append(kr)

            # 동의어에서 추가 후보 (영어 우선)
            for s in syns:
                s_lower = s.lower()
                if s_lower == token_lower:
                    continue
                if s_lower in query_lower:
                    continue
                # 영어 동의어 우선
                is_eng_syn = all(c < '\u0080' or c in ' -_/' for c in s)
                if is_eng_syn:
                    alternatives.insert(0, s)  # 영어를 앞에
                else:
                    alternatives.append(s)

            # 중복 제거 후 첫 번째 대체만 사용
            for alt in alternatives:
                if alt.lower() in query_lower:
                    continue
                variant = self._replace_token(query, match.token, alt)
                if variant and variant.lower() not in seen and variant.lower() != query_lower:
                    variants.append(variant)
                    seen.add(variant.lower())
                    break

        return variants[:max_variants]

    @staticmethod
    def _replace_token(query: str, token: str, replacement: str) -> str | None:
        """쿼리 내 토큰을 대체 문자열로 치환."""
        is_korean = any('\uac00' <= c <= '\ud7a3' for c in token)
        if is_korean:
            idx = query.lower().find(token.lower())
            if idx == -1:
                return None
            return query[:idx] + replacement + query[idx + len(token):]
        else:
            pattern = re.compile(rf"\b{re.escape(token)}\b", re.IGNORECASE)
            result = pattern.sub(replacement, query, count=1)
            return result if result != query else None


# 싱글톤 인스턴스
_instance: AbbreviationExpander | None = None


def get_abbreviation_expander(
    dict_path: str | Path = "data/semicon_word.json",
) -> AbbreviationExpander:
    """싱글톤 AbbreviationExpander 반환.

    semicon_word.json에서 런타임에 인덱스를 빌드한다.
    서버 재시작 시 사전 변경사항이 자동 반영됨.
    """
    global _instance
    if _instance is None:
        _instance = AbbreviationExpander.from_raw_dict(dict_path)
    return _instance


__all__ = [
    "AbbreviationExpander",
    "AbbreviationMatch",
    "ExpandResult",
    "get_abbreviation_expander",
]
