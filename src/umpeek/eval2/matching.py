from __future__ import annotations

import re
from collections import defaultdict
from functools import lru_cache
from typing import Any, Mapping, Sequence

from umpeek.attack_baselines.metrics import normalize_user_semantic_state

from .schema import AtomMatch, AtomParseResult, MetricAtom, USER_MODEL_ATOM_CATEGORIES, clone_json


_ALIASES = {
    "favourite": "favorite",
    "prefers": "prefer",
    "preferred": "prefer",
    "likes": "like",
    "liked": "like",
    "dislikes": "dislike",
    "can't": "cannot",
    "can not": "cannot",
    "tool state": "tool_state",
}

_GRAPH_RELATION_TUPLE_RE = re.compile(
    r"^\((?P<subject>[^,]+),(?P<predicate>[^,]+),(?P<object>.+)\)$"
)


@lru_cache(maxsize=200_000)
def _normalize_atom_text_cached(text: str) -> str:
    text = text.strip().lower()
    for source, target in _ALIASES.items():
        text = text.replace(source, target)
    text = re.sub(r"[`'\"“”‘’]", "", text)
    text = re.sub(r"[^a-z0-9_:=/|,(). -]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*([=:/|,])\s*", r"\1", text)
    return text.strip(" .;,")


def normalize_atom_text(value: Any) -> str:
    return _normalize_atom_text_cached(str(value or ""))


@lru_cache(maxsize=200_000)
def _tokenize(text: str) -> set[str]:
    return frozenset(token for token in re.split(r"[^a-z0-9_]+", normalize_atom_text(text)) if token)


@lru_cache(maxsize=200_000)
def _behavior_semantic_variants(text: str) -> set[str]:
    normalized = normalize_atom_text(text)
    variants = {normalized} if normalized else set()
    for prefix in ("personamemv2_choice=", "personalens_response=", "locomo_answer="):
        if normalized.startswith(prefix):
            variants.add(normalized[len(prefix) :].strip())
    if normalized.startswith("personalens_affinity="):
        stripped = normalized.split("=", 1)[1].strip()
        variants.add(stripped)
        if "|" in stripped:
            variants.add(stripped.split("|", 1)[1].strip())
    return {variant for variant in variants if variant}


def _graph_relation_parts(text: str) -> tuple[str, str, str] | None:
    match = _GRAPH_RELATION_TUPLE_RE.match(normalize_atom_text(text))
    if not match:
        return None
    return (
        match.group("subject").strip(),
        match.group("predicate").strip(),
        match.group("object").strip(),
    )


def _graph_relation_category(predicate: str) -> str | None:
    if re.search(r"(?:^|_)preference(?:_|$)", predicate):
        return "preferences"
    if re.search(r"(?:^|_)constraint(?:_|$)", predicate):
        return "constraints"
    if re.search(r"(?:^|_)fact(?:_|$)", predicate):
        return "facts"
    if re.search(r"(?:^|_)relation(?:_|$)", predicate):
        return "relations"
    return None


def _atom_categories_compatible(gold_atom: MetricAtom, recovered_atom: MetricAtom) -> bool:
    if gold_atom.category == recovered_atom.category:
        return True

    gold_relation = _graph_relation_parts(gold_atom.text)
    if gold_atom.category == "relations" and gold_relation is not None:
        _subject, predicate, _object_text = gold_relation
        if _graph_relation_category(predicate) == recovered_atom.category:
            return True

    recovered_relation = _graph_relation_parts(recovered_atom.text)
    if recovered_atom.category == "relations" and recovered_relation is not None:
        _subject, predicate, _object_text = recovered_relation
        if _graph_relation_category(predicate) == gold_atom.category:
            return True

    return False


def _atom_semantic_variants(text: str) -> set[str]:
    variants = set(_behavior_semantic_variants(text))
    relation = _graph_relation_parts(text)
    if relation is not None:
        _subject, _predicate, object_text = relation
        variants.update(_behavior_semantic_variants(object_text))
    return {variant for variant in variants if variant}


def _exact_lookup_variants(text: str) -> set[str]:
    normalized = normalize_atom_text(text)
    variants = {normalized} if normalized else set()
    variants.update(_behavior_semantic_variants(normalized))
    extra: set[str] = set()
    for variant in variants:
        if "|" in variant:
            extra.add(variant.split("|", 1)[1].strip())
    variants.update(item for item in extra if item)
    return {variant for variant in variants if variant}


def _semantic_group_from_text(text: str) -> str | None:
    tuple_match = re.match(r"^\(([^,]+),([^,]+),(.+)\)$", text)
    if tuple_match:
        subject, predicate, _object_text = tuple_match.groups()
        return f"{subject.strip()}::{predicate.strip()}"
    if "=" in text:
        return text.split("=", 1)[0].strip()
    return None


def _coerce_atom_dict(payload: Mapping[str, Any], *, sample_id: str, source: str) -> MetricAtom:
    category = str(payload.get("category") or payload.get("canonical_key") or "facts")
    if category not in USER_MODEL_ATOM_CATEGORIES:
        category = "facts"
    text = normalize_atom_text(
        payload.get("text")
        or payload.get("canonical_text")
        or payload.get("canonical_value")
        or payload.get("value")
    )
    metadata = clone_json(payload.get("metadata", {})) if isinstance(payload.get("metadata", {}), Mapping) else {}
    atom_type = str(payload.get("atom_type") or metadata.get("atom_type") or "semantic")
    semantic_group = payload.get("semantic_group") or metadata.get("semantic_group") or _semantic_group_from_text(text)
    return MetricAtom(
        category=category,
        text=text,
        sample_id=sample_id or str(payload.get("sample_id") or ""),
        atom_id=str(payload.get("atom_id") or ""),
        atom_type=atom_type,
        semantic_group=(None if semantic_group in (None, "") else str(semantic_group)),
        source=source,
        metadata=metadata,
    )


def atoms_from_user_model(payload: Any, *, sample_id: str = "", source: str = "unknown") -> AtomParseResult:
    if payload in (None, "", [], {}):
        return AtomParseResult(
            atoms=(),
            parse_status="empty",
            normalization_source="empty",
            parse_failed=False,
            excluded_count=0,
        )

    try:
        if isinstance(payload, Mapping) and isinstance(payload.get("atoms"), Sequence):
            atoms = tuple(
                _coerce_atom_dict(item, sample_id=sample_id, source=source)
                if isinstance(item, Mapping)
                else MetricAtom(category="facts", text=normalize_atom_text(item), sample_id=sample_id, source=source)
                for item in payload["atoms"]
                if item not in (None, "")
            )
            return AtomParseResult(
                atoms=atoms,
                parse_status="ok" if atoms else "empty",
                normalization_source="structured_atoms",
                parse_failed=False,
                excluded_count=0,
            )

        normalized = normalize_user_semantic_state(payload)
        atoms_list: list[MetricAtom] = []
        for category in USER_MODEL_ATOM_CATEGORIES:
            for item in normalized.get(category, []):
                text = normalize_atom_text(item)
                if not text:
                    continue
                atoms_list.append(
                    MetricAtom(
                        category=category,
                        text=text,
                        sample_id=sample_id,
                        atom_type="semantic",
                        semantic_group=_semantic_group_from_text(text),
                        source=source,
                    )
                )
        parse_failed = bool(normalized.get("parse_failed", False))
        parse_status = "parse_failed" if parse_failed and not atoms_list else "ok"
        if not atoms_list and not parse_failed:
            parse_status = "empty"
        return AtomParseResult(
            atoms=tuple(atoms_list),
            parse_status=parse_status,
            normalization_source=str(normalized.get("normalization_source") or "latent_user_model_v2"),
            parse_failed=parse_failed,
            excluded_count=int(normalized.get("excluded_count", 0) or 0),
            metadata={
                "excluded_by_scope": clone_json(normalized.get("excluded_by_scope", {})),
                "metric_scope_version": normalized.get("metric_scope_version"),
            },
        )
    except Exception as exc:
        return AtomParseResult(
            atoms=(),
            parse_status="parse_failed",
            normalization_source="exception",
            parse_failed=True,
            excluded_count=0,
            metadata={"error_type": exc.__class__.__name__, "error": str(exc)},
        )


def _alias_hit(gold_text: str, recovered_text: str, semantic_aliases: Mapping[str, Sequence[str]] | None) -> bool:
    if semantic_aliases is None:
        return False
    gold_normalized = normalize_atom_text(gold_text)
    recovered_normalized = normalize_atom_text(recovered_text)
    aliases = {normalize_atom_text(item) for item in semantic_aliases.get(gold_normalized, [])}
    aliases.update(
        key for key, values in semantic_aliases.items() if recovered_normalized in {normalize_atom_text(item) for item in values}
    )
    return recovered_normalized in aliases or gold_normalized in aliases


def semantic_atom_score(
    gold_atom: MetricAtom,
    recovered_atom: MetricAtom,
    *,
    semantic_aliases: Mapping[str, Sequence[str]] | None = None,
) -> float:
    if not _atom_categories_compatible(gold_atom, recovered_atom):
        return 0.0
    gold_text = normalize_atom_text(gold_atom.text)
    recovered_text = normalize_atom_text(recovered_atom.text)
    if gold_text == recovered_text:
        return 1.0
    if _alias_hit(gold_text, recovered_text, semantic_aliases):
        return 0.95
    best_score = 0.0
    for gold_variant in _atom_semantic_variants(gold_text):
        for recovered_variant in _atom_semantic_variants(recovered_text):
            if gold_variant and recovered_variant and (
                gold_variant in recovered_variant or recovered_variant in gold_variant
            ):
                best_score = max(best_score, 0.9)
                continue
            gold_tokens = _tokenize(gold_variant)
            recovered_tokens = _tokenize(recovered_variant)
            if not gold_tokens or not recovered_tokens:
                continue
            token_overlap = len(gold_tokens & recovered_tokens)
            token_union = len(gold_tokens | recovered_tokens)
            jaccard = token_overlap / token_union if token_union else 0.0
            recall_like = token_overlap / len(gold_tokens)
            best_score = max(best_score, jaccard, 0.85 * recall_like if recall_like >= 0.8 else 0.0)
    return best_score


def match_atoms(
    gold_atoms: Sequence[MetricAtom],
    recovered_atoms: Sequence[MetricAtom],
    *,
    semantic_aliases: Mapping[str, Sequence[str]] | None = None,
    semantic_threshold: float = 0.72,
) -> dict[str, Any]:
    matches: list[AtomMatch] = []
    used_gold_indexes: set[int] = set()
    used_recovered_indexes: set[int] = set()

    recovered_by_key: dict[tuple[str, str], list[int]] = defaultdict(list)
    for recovered_index, recovered_atom in enumerate(recovered_atoms):
        for lookup_text in _exact_lookup_variants(recovered_atom.text):
            recovered_by_key[(recovered_atom.category, lookup_text)].append(recovered_index)

    for gold_index, gold_atom in enumerate(gold_atoms):
        for lookup_text in _exact_lookup_variants(gold_atom.text):
            key = (gold_atom.category, lookup_text)
            matched = False
            for recovered_index in recovered_by_key.get(key, []):
                if recovered_index in used_recovered_indexes:
                    continue
                matches.append(AtomMatch(gold_atom=gold_atom, recovered_atom=recovered_atoms[recovered_index], match_type="exact", score=1.0))
                used_gold_indexes.add(gold_index)
                used_recovered_indexes.add(recovered_index)
                matched = True
                break
            if matched:
                break

    candidates: list[tuple[float, int, int]] = []
    for gold_index, gold_atom in enumerate(gold_atoms):
        if gold_index in used_gold_indexes:
            continue
        for recovered_index, recovered_atom in enumerate(recovered_atoms):
            if recovered_index in used_recovered_indexes:
                continue
            score = semantic_atom_score(gold_atom, recovered_atom, semantic_aliases=semantic_aliases)
            if semantic_threshold <= score:
                candidates.append((score, gold_index, recovered_index))

    for score, gold_index, recovered_index in sorted(candidates, reverse=True):
        if gold_index in used_gold_indexes or recovered_index in used_recovered_indexes:
            continue
        matches.append(
            AtomMatch(
                gold_atom=gold_atoms[gold_index],
                recovered_atom=recovered_atoms[recovered_index],
                match_type="semantic_alias_or_paraphrase",
                score=score,
            )
        )
        used_gold_indexes.add(gold_index)
        used_recovered_indexes.add(recovered_index)

    unmatched_gold = [atom for atom_index, atom in enumerate(gold_atoms) if atom_index not in used_gold_indexes]
    unmatched_recovered = [atom for atom_index, atom in enumerate(recovered_atoms) if atom_index not in used_recovered_indexes]
    return {
        "matches": matches,
        "matched_count": len(matches),
        "unmatched_gold": unmatched_gold,
        "unmatched_recovered": unmatched_recovered,
        "match_records": [match.to_dict() for match in matches],
        "unmatched_gold_atoms": [atom.to_dict() for atom in unmatched_gold],
        "unmatched_recovered_atoms": [atom.to_dict() for atom in unmatched_recovered],
    }
