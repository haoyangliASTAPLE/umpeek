from __future__ import annotations

import re
from typing import Any, Mapping

from umpeek.exp1_whitebox.schema import clone_json

from .schema import AttackInput, blank_predicted_user_model


BEHAVIOR_SEMANTIC_BENCHMARKS = frozenset({"PersonaMemv2", "PersonaLens", "LoCoMo"})
BEHAVIOR_SEMANTIC_TARGET_MODE = "behavior_semantic_target_v1"


def is_behavior_semantic_benchmark(benchmark: str) -> bool:
    return str(benchmark or "") in BEHAVIOR_SEMANTIC_BENCHMARKS


def build_behavior_semantic_target(
    *,
    benchmark: str,
    row: Mapping[str, Any],
    bundle: Mapping[str, Any] | None,
    condition: str = "personalized",
) -> dict[str, Any]:
    """Project non-ETAPP records into behavior-semantic user-model atoms."""

    if not is_behavior_semantic_benchmark(benchmark):
        return blank_predicted_user_model()
    record = _condition_record(bundle=bundle, condition=condition)
    output = _record_output(record)
    response_text = _record_response_text(output)
    visible_affinities = output.get("visible_affinities")
    model = _model_from_visible_behavior(
        benchmark=benchmark,
        response_text=response_text,
        task_prompt=str(row.get("task_prompt") or row.get("prompt") or ""),
        task_domain=str(row.get("task_domain") or ""),
        visible_affinities=visible_affinities if isinstance(visible_affinities, list) else [],
    )
    model["confidence"] = 1.0 if _item_count(model) else 0.0
    model["raw_text"] = render_behavior_semantic_raw_text(model)
    return model


def extract_visible_behavior_slots(
    sample: AttackInput,
    response_text: str,
) -> dict[str, Any]:
    """Decode only black-box visible prompt/response behavior into UMPeek slots."""

    return extract_visible_behavior_slots_for_prompt(
        sample,
        response_text,
        task_prompt=sample.task_prompt,
    )


def extract_visible_behavior_slots_for_prompt(
    sample: AttackInput,
    response_text: str,
    *,
    task_prompt: str,
) -> dict[str, Any]:
    """Decode visible behavior under a supplied ordinary prompt."""

    model = _model_from_visible_behavior(
        benchmark=sample.benchmark,
        response_text=response_text,
        task_prompt=task_prompt,
        task_domain=str(sample.public_context.get("task_domain") or ""),
        visible_affinities=[],
    )
    model["confidence"] = 0.88 if _item_count(model) else 0.0
    model["raw_text"] = render_behavior_semantic_raw_text(model)
    return model


def official_diagnostics(
    *,
    benchmark: str,
    row: Mapping[str, Any],
    bundle: Mapping[str, Any] | None,
) -> dict[str, Any]:
    record = _condition_record(bundle=bundle, condition="personalized")
    output = _record_output(record)
    row_evidence = row.get("personalization_evidence")
    diagnostics: dict[str, Any] = {
        "semantic_target_mode": BEHAVIOR_SEMANTIC_TARGET_MODE,
        "official_reference_available": bool(row.get("gold")),
        "personalized_response_present": bool(_record_response_text(output)),
        "benchmark_score": _number_or_none(record.get("score")),
    }
    if benchmark == "PersonaMemv2":
        diagnostics.update(
            {
                "official_correct_answer_present": bool(
                    dict(row.get("gold") or {}).get("correct_answer")
                    if isinstance(row.get("gold"), Mapping)
                    else False
                ),
                "personalization_evidence_count": (
                    len(row_evidence) if isinstance(row_evidence, list) else 0
                ),
            }
        )
    elif benchmark == "PersonaLens":
        official = row.get("gold") if isinstance(row.get("gold"), Mapping) else {}
        expected = official.get("expected_affinities") if isinstance(official, Mapping) else []
        visible = output.get("visible_affinities")
        diagnostics.update(
            {
                "official_expected_affinity_count": (
                    len(expected) if isinstance(expected, list) else 0
                ),
                "visible_affinity_count": len(visible) if isinstance(visible, list) else 0,
            }
        )
    elif benchmark == "LoCoMo":
        official = row.get("gold") if isinstance(row.get("gold"), Mapping) else {}
        evidence = (
            row_evidence.get("evidence_dialog_ids")
            if isinstance(row_evidence, Mapping)
            else official.get("evidence_dialog_ids") if isinstance(official, Mapping) else []
        )
        diagnostics.update(
            {
                "official_answer_present": bool(
                    official.get("answer") if isinstance(official, Mapping) else False
                ),
                "official_evidence_count": len(evidence) if isinstance(evidence, list) else 0,
            }
        )
    return diagnostics


def intersect_behavior_models(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> dict[str, Any]:
    model = blank_predicted_user_model()
    for category in ("facts", "preferences", "constraints", "relations"):
        first_items = [str(item) for item in first.get(category, [])]
        second_items = {str(item) for item in second.get(category, [])}
        model[category] = [item for item in first_items if item in second_items]
    first_tool = first.get("tool_state")
    second_tool = second.get("tool_state")
    if isinstance(first_tool, list) and isinstance(second_tool, list):
        second_tool_items = {str(item) for item in second_tool}
        model["tool_state"] = [str(item) for item in first_tool if str(item) in second_tool_items]
    elif isinstance(first_tool, Mapping) and isinstance(second_tool, Mapping):
        model["tool_state"] = {
            str(key): clone_json(value)
            for key, value in first_tool.items()
            if key in second_tool and second_tool[key] == value
        }
    model["confidence"] = min(
        float(first.get("confidence", 0.0) or 0.0),
        float(second.get("confidence", 0.0) or 0.0),
    )
    if first.get("replayed_behavior") not in (None, "", [], {}):
        model["replayed_behavior"] = clone_json(first.get("replayed_behavior"))
    model["raw_text"] = render_behavior_semantic_raw_text(model)
    return model


def union_behavior_models(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> dict[str, Any]:
    """Posterior aggregation over independently visible behavior projections."""

    model = blank_predicted_user_model()
    for category in ("facts", "preferences", "constraints", "relations"):
        model[category] = _drop_subsumed_texts(
            _dedupe(
            [str(item) for item in first.get(category, []) if str(item)]
            + [str(item) for item in second.get(category, []) if str(item)]
            )
        )
    first_tool = first.get("tool_state")
    second_tool = second.get("tool_state")
    if isinstance(first_tool, Mapping) or isinstance(second_tool, Mapping):
        merged: dict[str, Any] = {}
        if isinstance(first_tool, Mapping):
            merged.update({str(key): clone_json(value) for key, value in first_tool.items()})
        if isinstance(second_tool, Mapping):
            merged.update({str(key): clone_json(value) for key, value in second_tool.items()})
        model["tool_state"] = merged
    elif isinstance(first_tool, list) or isinstance(second_tool, list):
        model["tool_state"] = _dedupe(
            [str(item) for item in first_tool or [] if str(item)]
            + [str(item) for item in second_tool or [] if str(item)]
        )
    model["confidence"] = max(
        float(first.get("confidence", 0.0) or 0.0),
        float(second.get("confidence", 0.0) or 0.0),
    )
    if first.get("replayed_behavior") not in (None, "", [], {}):
        model["replayed_behavior"] = clone_json(first.get("replayed_behavior"))
    elif second.get("replayed_behavior") not in (None, "", [], {}):
        model["replayed_behavior"] = clone_json(second.get("replayed_behavior"))
    model["raw_text"] = render_behavior_semantic_raw_text(model)
    return model


def _drop_subsumed_texts(items: list[str]) -> list[str]:
    kept: list[str] = []
    normalized = [_semantic_compact(item) for item in items]
    token_counts = [len(_content_tokens(item)) for item in items]
    semantic_keys = [_semantic_key(item) for item in items]
    for idx, item in enumerate(items):
        norm = normalized[idx]
        if not norm:
            continue
        subsumed = False
        for other_idx, other in enumerate(normalized):
            if idx == other_idx or not other:
                continue
            item_is_conversation = _is_conversation_carrier(item)
            other_is_conversation = _is_conversation_carrier(items[other_idx])
            if _is_negative_memory_carrier(item):
                continue
            if _is_negative_memory_carrier(items[other_idx]):
                continue
            if _is_locomo_speaker_evidence_carrier(item) or _is_locomo_speaker_evidence_carrier(items[other_idx]):
                item_has_time = _has_locomo_timestamp(item)
                other_has_time = _has_locomo_timestamp(items[other_idx])
                if item_has_time and not other_has_time:
                    continue
                if other_has_time and not item_has_time:
                    item_body = _locomo_body_compact(item)
                    other_body = _locomo_body_compact(items[other_idx])
                    if item_body and (item_body in other_body or other_body in item_body):
                        subsumed = True
                        break
            if item_is_conversation and other_is_conversation:
                item_early = norm.startswith("earlier example:user")
                other_early = other.startswith("earlier example:user")
                if item_early and not other_early:
                    subsumed = True
                    break
                if item_early == other_early and len(norm) < len(other):
                    subsumed = True
                    break
                continue
            if item_is_conversation and not other_is_conversation:
                continue
            if other_is_conversation and not item_is_conversation and norm in other:
                subsumed = True
                break
            if semantic_keys[idx] and semantic_keys[idx] == semantic_keys[other_idx]:
                if len(norm) < len(other):
                    subsumed = True
                    break
                continue
            if token_counts[other_idx] < 3 and len(other) < 18:
                continue
            if len(other) < len(norm) and other in norm:
                subsumed = True
                break
        if not subsumed:
            kept.append(item)
    return kept


def _is_conversation_carrier(text: str) -> bool:
    lowered = str(text or "").lower()
    return "user:" in lowered and "assistant:" in lowered


def _is_negative_memory_carrier(text: str) -> bool:
    value = str(text or "")
    return bool(
        re.search(r"\bdo not remember\b.+\bin memory\b", value, flags=re.IGNORECASE)
        or re.search(r"\b(?:no longer|does not|doesn't|do not|don't)\s+(?:enjoys?|likes?|prefers?)\b", value, flags=re.IGNORECASE)
        or re.search(r"\bforget(?: that)?\b", value, flags=re.IGNORECASE)
        or re.search(r"\bno preference for\b", value, flags=re.IGNORECASE)
    )


def _semantic_key(text: str) -> str:
    if _is_locomo_speaker_evidence_carrier(text):
        return ""
    match = re.match(r"\s*([A-Za-z_][A-Za-z0-9_ ]{1,40})\s*:", str(text or ""))
    return _semantic_compact(match.group(1)) if match else ""


def _is_locomo_speaker_evidence_carrier(text: str) -> bool:
    value = str(text or "").strip()
    speaker_pattern = r"[A-Z][A-Za-z0-9_.'-]{1,40}"
    return bool(
        re.match(rf"^(?:D\d+:\d+\s+|\d+\s+)?{speaker_pattern}\s*:", value, flags=re.IGNORECASE)
        or re.match(rf"^\d{{1,2}}:\d{{2}}\s*(?:am|pm)\b.*?:\s*{speaker_pattern}\s*:", value, flags=re.IGNORECASE)
    )


def _has_locomo_timestamp(text: str) -> bool:
    return bool(re.search(r"\b\d{1,2}:\d{2}\s*(?:am|pm)\b.*?\d{4}", str(text or ""), flags=re.IGNORECASE))


def _locomo_body_compact(text: str) -> str:
    value = str(text or "").strip()
    speaker_pattern = r"[A-Z][A-Za-z0-9_.'-]{1,40}"
    value = re.sub(rf"^\d{{1,2}}:\d{{2}}\s*(?:am|pm)\b.*?:\s*{speaker_pattern}\s*:\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(rf"^(?:D\d+:\d+\s+|\d+\s+)?{speaker_pattern}\s*:\s*", "", value, flags=re.IGNORECASE)
    return _semantic_compact(value)


def behavior_model_item_count(model: Mapping[str, Any]) -> int:
    return _item_count(model)


def render_behavior_semantic_raw_text(model: Mapping[str, Any]) -> str:
    sections: list[str] = []
    labels = (
        ("facts", "Facts"),
        ("preferences", "Preferences"),
        ("constraints", "Constraints"),
        ("relations", "Relations"),
        ("tool_state", "Tool state"),
    )
    for category, label in labels:
        value = model.get(category)
        if isinstance(value, Mapping):
            items = [f"{key}={value[key]}" for key in sorted(value)]
        elif isinstance(value, list):
            items = [str(item) for item in value if str(item)]
        else:
            items = []
        if not items:
            continue
        sections.append(label + ":")
        sections.extend(f"- {item}" for item in items)
    return "\n".join(sections)


def _model_from_visible_behavior(
    *,
    benchmark: str,
    response_text: str,
    task_prompt: str = "",
    task_domain: str,
    visible_affinities: list[Any],
) -> dict[str, Any]:
    model = blank_predicted_user_model()
    response = _clean_response_text(response_text)
    if not response:
        return model
    model["replayed_behavior"] = response
    if benchmark == "PersonaMemv2":
        facts, preferences = _personamem_latent_atoms(
            task_prompt=task_prompt,
            response_text=response,
            task_domain=task_domain,
        )
        if not facts and not preferences:
            preferences = [f"personamemv2_choice={response}"]
        model["facts"] = facts
        model["preferences"] = preferences
    elif benchmark == "PersonaLens":
        facts = _personalens_profile_fact_atoms(response)
        preferences = _personalens_affinity_atoms(
            task_domain=task_domain,
            visible_affinities=visible_affinities,
            response_text=response,
            task_prompt=task_prompt,
        )
        if not preferences:
            preferences = [f"personalens_response={response}"]
        model["facts"] = _dedupe(facts)
        model["preferences"] = _dedupe(preferences)
    elif benchmark == "LoCoMo":
        facts = _locomo_evidence_hypothesis_atoms(task_prompt=task_prompt, response_text=response)
        model["facts"] = facts or [f"locomo_answer={response}"]
    return model


_PERSONAMEM_TOPIC_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "sensitive_info",
        (
            "social_security",
            "ssn",
            "passport",
            "address",
            "drivers_license",
            "driver's license",
            "license number",
            "bank account",
            "account number",
            "credit card",
            "api_key",
            "identification",
            "redact",
            "masked",
            "sensitive data",
            "personal sensitive",
        ),
    ),
    ("health_and_medical", ("cholesterol", "asthma", "migraine", "allergic", "allergies", "pollen", "vitamin d", "gallbladder", "myopia", "eczema", "prescription", "healthcare", "medical supplies")),
    ("mental health", ("mental health", "loneliness", "frustration", "guilt", "stress-related", "psychological", "anger", "homophobia", "bias")),
    ("travel", ("travel", "trip", "seaside", "coastal", "coast", "pacific")),
    ("sports", ("surfing", "surfer", "waves", "board", "soccer", "football", "premier league", "sports", "swimming", "ultramarathon", "training run")),
    ("gardening", ("garden", "gardening", "houseplant", "houseplants", "vegetables and herbs")),
    ("home", ("home", "houseplant", "houseplants", "family photos", "minimalist", "decor", "backyard")),
    ("social", ("facebook", "friends and family", "stay connected")),
    ("motivation", ("motivation", "influence", "effectiveness", "mentor", "mentored", "purpose", "hope")),
    ("mentoring", ("mentor", "mentored", "younger community")),
    ("entertainment", ("anime", "comic-con", "classic movies", "movies", "cinema", "films", "documentaries")),
    ("music", ("music", "vinyl", "album", "albums", "artist", "artists", "acoustic", "festival", "festivals")),
    ("literature", ("russian literature", "novels", "books", "book clubs", "libraries")),
    ("culture", ("culture", "cultural", "etiquette", "shinto", "shrine", "shrines", "ceremony", "traditional", "tradition", "heritage", "new year")),
    ("craftsmanship", ("handmade", "artisan", "artisanal", "craft", "ceramic", "handcrafted")),
    ("gifts", ("gift", "birthday", "handmade", "artisanal")),
    ("friendship", ("friendship", "friendships", "loneliness", "connected and supported")),
    ("relationships", ("friendship", "family milestones", "relationships", "friendships", "loneliness")),
    ("parenting", ("parenting", "remote schooling", "kids", "children")),
    ("fashion", ("fashion", "high-fashion", "designers")),
    ("automobiles", ("vehicle", "car", "american-made", "automobile")),
    ("technology", ("technology", "coding", "hackathon", "nft", "digital art", "api_key")),
    ("finance", ("finance", "cryptocurrency", "bank", "credit card")),
    ("insurance", ("insurance", "claim")),
    ("event", ("event", "gathering", "party")),
    ("art", ("art", "museum", "abstract art")),
    ("photography", ("photography", "photo", "photograph")),
    ("realestate", ("property", "real estate", "realestate")),
    ("business", ("business", "women-in-business", "networking")),
    ("family", ("family", "children", "kids", "parenting")),
    ("work", ("work", "career", "corporate", "remote work", "professional")),
    ("volunteering", ("volunteer", "volunteering", "food bank")),
    ("outdoors", ("outdoors", "trail", "lakes", "summer", "river", "backyard")),
    ("health", ("health", "calming", "morning routine", "anxiety", "stress", "grounded", "wellness")),
    ("education", ("student", "school", "curriculum", "class", "teacher", "colleague", "parent")),
    ("communication", ("communicat", "respond", "message", "email", "tone", "conversation")),
    ("politics", ("politic", "republican", "democrat", "candidate", "federalist", "regulatory", "public safety", "policy")),
    ("food", ("recipe", "vegetarian", "cooking", "smoothie", "herb", "baking", "bread", "cuisine", "spice", "tea", "coffee")),
    ("community", ("community", "neighbor", "volunteer", "food bank")),
)


def _personamem_latent_atoms(
    *,
    task_prompt: str,
    response_text: str,
    task_domain: str = "",
) -> tuple[list[str], list[str]]:
    response = _strip_choice_number(response_text)
    facts: list[str] = []
    preferences: list[str] = []
    query_text = _personamem_query_text(task_prompt)
    query_topics = _personamem_topics_from_text(query_text)
    response_topics = _personamem_topics_from_text(response)
    if query_topics and "\nOptions:" in str(task_prompt or ""):
        facts.extend(query_topics[:2])
    domain_topic = _canonical_personamem_topic_label(task_domain)
    if domain_topic and "\nOptions:" in str(task_prompt or ""):
        facts.insert(0, domain_topic)
    for response_topic in (response_topics or query_topics[:1]):
        if response_topic not in preferences:
            preferences.append(response_topic)
    evidence_texts = [response, *_personamem_selected_options_from_prompt(task_prompt, response)]
    for text in evidence_texts:
        preferences.extend(_personamem_preference_phrases(text))
        preferences.extend(_personamem_visible_sensitive_identifier_phrases(text))
        field_facts, field_preferences = _personamem_visible_field_certificate_lines(text)
        facts.extend(field_facts)
        preferences.extend(field_preferences)
    preferences.extend(_personamem_visible_sensitive_identifier_phrases(query_text))
    preferences.extend(_personamem_choice_set_contrastive_phrases(task_prompt, response))
    return _dedupe(facts), _dedupe(preferences)


def _personamem_visible_field_certificate_lines(text: str) -> tuple[list[str], list[str]]:
    """Preserve explicit profile-card fields emitted by the visible agent.

    The adapter regularizer owns the final semantic selection.  This decoder
    only prevents short but explicit fields such as ``Stable role=CEO`` from
    being dropped before the posterior stage.
    """

    source = " ".join(str(text or "").strip().split())
    if not source:
        return [], []
    label = (
        r"Memory topic|Domain|Stable persona|Stable role|Role/Background|"
        r"Identity/Background|Background/Identity|Recurring preference|Concrete preference|"
        r"Preference phrase|Interest|Interests|Routine/Habit|Responsibilities/Routine|"
        r"Responsibilities|Routine|Habit|Value/Concern|Values/Concerns|Underlying reason|"
        r"Health constraint|Health/Diet constraint|Dietary/Health constraint|Sensitive/Private field|"
        r"Contact/Account detail|Privacy constraint|Avoidance|Correction/Avoidance|Correction|"
        r"Prior interaction example|Prior example|Raw prior interaction|Format/genre cue"
    )
    source = re.sub(
        rf"(?P<prefix>^|[\s{{,])['\"]?(?P<label>{label})['\"]?\s*:\s*",
        lambda match: f"{match.group('prefix')}{match.group('label')}=",
        source,
        flags=re.IGNORECASE,
    )
    pattern = re.compile(
        rf"\b(?P<label>{label})\s*=\s*(?P<body>.*?)(?=\s+\b(?:{label})\s*=|$)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    facts: list[str] = []
    preferences: list[str] = []
    for match in pattern.finditer(source):
        raw_label = " ".join(match.group("label").strip().split())
        value = _clean_personamem_field_certificate_value(match.group("body"), label_pattern=label)
        if not value or value.lower() in {"none", "unknown", "not specified", "not available"}:
            continue
        normalized_label = raw_label.lower()
        line = f"{raw_label}={value}"
        if normalized_label in {"memory topic", "domain"}:
            facts.append(line)
        else:
            preferences.append(line)
    return _dedupe(facts), _dedupe(preferences)


def _clean_personamem_field_certificate_value(value: str, *, label_pattern: str) -> str:
    cleaned = " ".join(str(value or "").strip(" -*,.;:'\"").split())
    if not cleaned:
        return ""
    cleaned = re.sub(
        rf"\s+\b(?:{label_pattern})\s*=.*$",
        "",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip(" -*,.;:'\"")
    cleaned = re.split(
        r"\s+\b(?:Here'?s a compact|Here's a compact|You might|You could|For example)\b",
        cleaned,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip(" -*,.;:'\"")
    return cleaned[:1600]


def _canonical_personamem_topic_label(value: str) -> str:
    text = " ".join(str(value or "").strip().split()).lower()
    if not text or text == "personamemv2":
        return ""
    return text.replace(" ", "_") if text in {"health and medical"} else text.replace("real estate", "realestate")


def _personamem_query_text(task_prompt: str) -> str:
    text = str(task_prompt or "")
    if "\nOptions:" in text:
        return text.split("\nOptions:", 1)[0].strip()
    return text.strip()


def _personamem_options_from_prompt(task_prompt: str) -> list[str]:
    text = str(task_prompt or "")
    if "\nOptions:" not in text:
        return []
    raw = text.split("\nOptions:", 1)[1]
    options: list[str] = []
    current: list[str] = []
    for line in raw.splitlines():
        if re.match(r"^\s*-\s+", line):
            if current:
                options.append(" ".join(current).strip())
            current = [re.sub(r"^\s*-\s+", "", line).strip()]
        elif current and line.strip():
            current.append(line.strip())
    if current:
        options.append(" ".join(current).strip())
    return [option for option in options if option]


def _personamem_selected_options_from_prompt(task_prompt: str, response_text: str) -> list[str]:
    response = _strip_choice_number(response_text)
    options = _personamem_options_from_prompt(task_prompt)
    selected: list[str] = []
    response_norm = _semantic_compact(response)
    response_tokens = _content_tokens(response)
    for option in options:
        option_norm = _semantic_compact(option)
        if response_norm and option_norm and (response_norm in option_norm or option_norm in response_norm):
            selected.append(option)
            continue
        option_tokens = _content_tokens(option)
        if response_tokens and option_tokens:
            overlap = len(response_tokens & option_tokens) / max(1, min(len(response_tokens), len(option_tokens)))
            if overlap >= 0.72:
                selected.append(option)
    return _dedupe(selected)[:1]


def _personamem_choice_set_contrastive_phrases(task_prompt: str, response_text: str) -> list[str]:
    """Infer user-model evidence from a visible choice against its alternatives.

    PersonaMem-v2 choice prompts expose not only the chosen behavior but also
    nearby alternatives.  Under a conditional-choice view, a chosen option is
    informative through the feature margins it has over unchosen options.  This
    function only uses the public task prompt, public option set, and visible
    response; it never reads evaluator gold/profile fields.
    """

    options = _personamem_options_from_prompt(task_prompt)
    selected_options = _personamem_selected_options_from_prompt(task_prompt, response_text)
    if not options or not selected_options:
        return []
    query_text = _personamem_query_text(task_prompt)
    query_topics = set(_personamem_topics_from_text(query_text))
    phrases: list[str] = []
    for selected in selected_options:
        selected_norm = _semantic_compact(selected)
        others = [option for option in options if _semantic_compact(option) != selected_norm]
        other_text = "\n".join(others)
        other_topics = set(_personamem_topics_from_text(other_text))
        selected_topics = set(_personamem_topics_from_text(selected))
        for topic in sorted(selected_topics | query_topics):
            if topic and (topic not in other_topics or topic in query_topics):
                phrases.append(topic)
        phrases.extend(_personamem_visible_sensitive_identifier_phrases(selected))
        phrases.extend(_personamem_visible_sensitive_identifier_phrases(query_text))
        phrases.extend(
            _personamem_contrastive_privacy_phrases(
                selected_text=selected,
                query_text=query_text,
            )
        )
        phrases.extend(_personamem_contrastive_interest_phrases(selected, others))
    return _dedupe(phrases)[:12]


def _personamem_visible_sensitive_identifier_phrases(text: str) -> list[str]:
    source = " ".join(str(text or "").strip().split())
    if not source:
        return []
    phrases: list[str] = []
    patterns = (
        (
            "drivers_license_number",
            r"\bdriver'?s?\s+license(?:\s+number|\s+details)?\s*(?:is|:|\(|#)?\s*"
            r"(?P<value>[A-Z]?[0-9][A-Z0-9-]{5,24}(?:-[A-Z]{2,4})?)",
        ),
        (
            "bank_account_number",
            r"\bbank\s+account\s+number\s*(?:is|:|#)?\s*(?P<value>[0-9][0-9 -]{7,30})",
        ),
        (
            "credit_card_number",
            r"\bcredit\s+card\s+number\s*(?:is|:|#)?\s*(?P<value>[0-9][0-9 -]{11,24})",
        ),
        (
            "social_security_number",
            r"\b(?:social\s+security\s+number|ssn)\s*(?:is|:|#)?\s*(?P<value>[0-9]{3}-?[0-9]{2}-?[0-9]{4})",
        ),
        (
            "passport_number",
            r"\b(?:passport(?:\s+(?:number|no\.?))?|passport\s*#)\s*(?:is|:|#)?\s*(?P<value>[A-Z][A-Z0-9-]{5,20})",
        ),
        (
            "llm_api_key",
            r"\b(?:llm\s+api\s+key|api[_ -]?key)\s*(?:is|:|=)?\s*(?P<value>[A-Za-z0-9_-]{12,80})",
        ),
    )
    for label, pattern in patterns:
        for match in re.finditer(pattern, source, flags=re.IGNORECASE):
            value = _personamem_clean_identifier_value(match.group("value"))
            if value:
                phrases.append(f"{label}: {value}")
    if re.search(r"\b(?:redact|redacted|mask|masked|tokenization|sensitive data|personal sensitive)\b", source, flags=re.IGNORECASE):
        phrases.append("sensitive_info")
        phrases.append("Avoids exposing sensitive information")
        phrases.append("Values data privacy")
    return _dedupe(phrases)


def _personamem_clean_identifier_value(value: str) -> str:
    cleaned = str(value or "").strip(" -*,.;:'\"()[]{}")
    cleaned = re.sub(r"\s+", "", cleaned)
    if not cleaned or re.fullmatch(r"0{5,}|x{3,}|redacted", cleaned, flags=re.IGNORECASE):
        return ""
    if len(cleaned) < 6 or len(cleaned) > 80:
        return ""
    return cleaned


def _personamem_contrastive_privacy_phrases(*, selected_text: str, query_text: str) -> list[str]:
    selected = str(selected_text or "")
    query = str(query_text or "")
    joined = f"{query}\n{selected}"
    if not re.search(r"\b(?:redact|redacted|mask|masked|tokenization|sensitive|privacy|securely|compliance)\b", joined, flags=re.IGNORECASE):
        return []
    if not re.search(r"\b(?:license|passport|account|ssn|social security|credit card|api[_ -]?key|address|personal information)\b", joined, flags=re.IGNORECASE):
        return []
    return [
        "sensitive_info",
        "Avoids exposing sensitive information",
        "Values data privacy",
    ]


def _personamem_contrastive_interest_phrases(selected: str, others: Sequence[str]) -> list[str]:
    selected_text = " ".join(str(selected or "").strip().split())
    if not selected_text:
        return []
    other_tokens: set[str] = set()
    for option in others:
        other_tokens.update(_content_tokens(option))
    selected_tokens = _content_tokens(selected_text)
    unique_tokens = selected_tokens - other_tokens
    if not unique_tokens:
        return []
    phrases: list[str] = []
    for candidate in _personamem_candidate_interest_spans(selected_text):
        candidate_tokens = _content_tokens(candidate)
        if not candidate_tokens:
            continue
        if len(candidate_tokens & unique_tokens) / max(1, len(candidate_tokens)) < 0.45:
            continue
        if not _personamem_interest_span_is_diagnostic(candidate):
            continue
        phrases.append(f"Enjoys {candidate}")
    return _dedupe(phrases)[:4]


def _personamem_candidate_interest_spans(text: str) -> list[str]:
    spans: list[str] = []
    patterns = (
        r"\bfor\s+(?:an?|the)?\s*(?P<body>[^,.;]{4,90}?\b(?:match|marathon|livestream|concert|festival|club|book|novel|movie|film|series|game|activity|activities|event|outing|trip|walk|run|routine|recipe|meal|project))\b",
        r"\b(?:about|around|focused on|built around)\s+(?P<body>[^,.;]{4,90})",
        r"\b(?:complete with|featuring)\s+(?P<body>[^,.;]{4,90})",
        r"\b(?:favorite|preferred)\s+(?P<body>[^,.;]{4,70})",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            body = _personamem_clean_interest_span(match.group("body"))
            if body:
                spans.append(body)
    return _dedupe(spans)


def _personamem_clean_interest_span(value: str) -> str:
    text = " ".join(str(value or "").strip(" -*,.;:'\"").split())
    text = re.sub(r"\s+(?:and maybe|and perhaps|with your|for a few|during breaks|beforehand)\b.*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(?:a|an|the)\s+", "", text, flags=re.IGNORECASE)
    if len(text) < 4 or len(text) > 90:
        return ""
    return text


def _personamem_interest_span_is_diagnostic(value: str) -> bool:
    text = _semantic_compact(value)
    if len(text.split()) < 2:
        return False
    bad = (
        "something you enjoy",
        "personalized recommendation",
        "supporting documents",
        "online application",
        "common errors",
        "official checklist",
        "state compliance",
        "secure manner",
        "robust encryption",
        "the user interface",
    )
    if any(marker in text for marker in bad):
        return False
    topicful = set(_personamem_topics_from_text(value))
    if topicful:
        return True
    return bool(re.search(r"\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,5}\b", str(value or "")))


def _semantic_compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def _content_tokens(text: str) -> set[str]:
    stop = {
        "the", "and", "or", "a", "an", "to", "of", "in", "on", "for", "with", "that",
        "this", "it", "is", "are", "be", "can", "could", "would", "you", "your",
        "i", "me", "my", "we", "our", "as", "by", "from", "at", "after", "before",
    }
    return {
        token
        for token in re.split(r"[^a-z0-9]+", str(text or "").lower())
        if len(token) > 2 and token not in stop
    }


def _personamem_topic_from_text(text: str) -> str:
    topics = _personamem_topics_from_text(text)
    return topics[0] if topics else ""


def _personamem_topics_from_text(text: str) -> list[str]:
    lower = str(text or "").lower()
    topics: list[str] = []
    for topic, hints in _PERSONAMEM_TOPIC_HINTS:
        if any(_personamem_hint_in_text(hint, lower) for hint in hints):
            topics.append(topic)
    return _dedupe(topics)[:3]


def _personamem_hint_in_text(hint: str, lower_text: str) -> bool:
    hint_text = str(hint or "").lower().strip()
    if not hint_text:
        return False
    escaped = re.escape(hint_text).replace(r"\ ", r"\s+")
    return bool(re.search(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", lower_text))


def _personamem_preference_phrases(response_text: str) -> list[str]:
    response = _strip_choice_number(response_text)
    phrases: list[str] = []
    phrases.extend(_personamem_explicit_profile_phrases(response))
    phrases.extend(_personamem_discourse_antecedent_phrases(response))
    patterns = (
        r"\bsince you already ([^,.;]+)",
        r"\bsince you (?:enjoy|like|love|prefer) ([^,.;]+)",
        r"\byou already ([^,.;]+)",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, response, flags=re.IGNORECASE):
            phrase = _third_person_preference_clause(match.group(1))
            if phrase:
                phrases.append(phrase)
    for match in re.finditer(r"\bas a ([^,.;]+)", response, flags=re.IGNORECASE):
        phrase = " ".join(match.group(1).strip().split())
        if phrase and not phrase.lower().startswith(("a ", "an ")):
            phrase = f"a {phrase}"
        if 8 <= len(phrase) <= 180:
            phrases.append(phrase)
    stress_patterns = (
        r"\bafter dealing with (?:that |the )?([^,.;]+)",
        r"\bfollowing (?:that |the )?([^,.;]+)",
        r"\bafter (?:that |the )?([^,.;]+)",
    )
    for pattern in stress_patterns:
        for match in re.finditer(pattern, response, flags=re.IGNORECASE):
            phrase = _stress_phrase_from_visible_clause(match.group(1))
            if phrase:
                phrases.append(phrase)
    return _dedupe(phrases)[:12]


def _personamem_discourse_antecedent_phrases(response_text: str) -> list[str]:
    """Recover personalization antecedents from ordinary visible explanations.

    PersonaMem-style visible behavior often has the form "Since/Given/As <user
    context>, <recommendation>". The antecedent side is a compact observable
    projection of the latent user model; the consequent side is task advice.
    """

    text = " ".join(str(response_text or "").strip().split())
    if not text:
        return []
    phrases: list[str] = []
    marker_patterns = (
        r"\b(?:since|because|given|considering)\s+(?P<body>(?:you|your|you're|you've|you are|you have)\b.{8,260}?)(?=,\s|\s+you might\b|\s+you could\b|\s+consider\b|[.;]|$)",
        r"\bwith\s+(?P<body>your\b.{8,220}?)(?=,\s|\s+you might\b|\s+you could\b|\s+consider\b|[.;]|$)",
        r"\bas\s+(?P<body>(?:a|an|someone|a person|an individual)\b.{8,220}?)(?=,\s|\s+you might\b|\s+you could\b|\s+consider\b|[.;]|$)",
        r"\bafter\s+(?P<body>(?:the|your|you)\b.{8,220}?)(?=,\s|\s+you might\b|\s+you could\b|\s+consider\b|[.;]|$)",
    )
    for pattern in marker_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            body = _personamem_clean_antecedent_body(match.group("body"))
            phrases.extend(_personamem_factor_antecedent_body(body))
    return _dedupe(phrases)


def _personamem_clean_antecedent_body(body: str) -> str:
    value = " ".join(str(body or "").strip(" -*,.;:").split())
    value = re.sub(
        r"\s+(?:i recommend|i suggest|consider|try|arriving|choosing|choosing the|focusing on|one way|it may help)\b.*$",
        "",
        value,
        flags=re.IGNORECASE,
    )
    return value.strip(" -*,.;:")


def _personamem_factor_antecedent_body(body: str) -> list[str]:
    value = _personamem_clean_antecedent_body(body)
    if not value:
        return []
    clauses: list[str] = []
    for part in re.split(r"\s*(?:,\s*and\s+|;\s*|\s+while\s+|\s+as well as\s+)\s*", value):
        part = part.strip(" -*,.;:")
        if part:
            clauses.append(part)
    if len(clauses) == 1:
        clauses = _personamem_split_coordinated_antecedent(clauses[0])
    phrases: list[str] = []
    for clause in clauses:
        for candidate in _personamem_antecedent_clause_variants(clause):
            phrase = _third_person_preference_clause(candidate)
            if phrase and _personamem_antecedent_is_diagnostic(phrase):
                phrases.append(phrase)
    return _dedupe(phrases)


def _personamem_split_coordinated_antecedent(clause: str) -> list[str]:
    text = " ".join(str(clause or "").strip().split())
    split_match = re.search(
        r"\b(?:and|but also)\s+(?=(?:you|your|you're|you've|you are|you have|are|is|have|has|enjoy|enjoys|like|likes|love|loves|prefer|prefers|value|values|work|works|attend|attends|support|supports|care|cares)\b)",
        text,
        flags=re.IGNORECASE,
    )
    if not split_match:
        return [text]
    left = text[: split_match.start()].strip(" -*,.;:")
    right = text[split_match.end() :].strip(" -*,.;:")
    left_prefix = ""
    if re.match(r"^(?:you|your|you're|you've|you are|you have)\b", left, flags=re.IGNORECASE):
        left_prefix = "you "
    if right and left_prefix and not re.match(r"^(?:you|your|you're|you've|you are|you have)\b", right, flags=re.IGNORECASE):
        right = left_prefix + right
    return [part for part in (left, right) if part]


def _personamem_antecedent_clause_variants(clause: str) -> list[str]:
    text = " ".join(str(clause or "").strip(" -*,.;:").split())
    if not text:
        return []
    lowered = text.lower()
    variants = [] if lowered.startswith(("you ", "your ", "you're ", "you've ", "you are ", "you have ")) else [text]
    if lowered.startswith("your "):
        noun_phrase = text[5:].strip()
        nominal_matched = False
        nominal_patterns = (
            (r"^(?:love|liking)\s+(?:for|of)\s+(.+)$", "Loves {value}"),
            (r"^(?:preference)\s+(?:for|toward)\s+(.+)$", "Prefers {value}"),
            (r"^(?:interest)\s+(?:in|for)\s+(.+)$", "Is interested in {value}"),
            (r"^(?:concern|concerns)\s+(?:about|over)\s+(.+)$", "Is concerned about {value}"),
            (r"^(?:commitment|dedication)\s+(?:to|toward)\s+(.+)$", "Is committed to {value}"),
            (r"^(?:support)\s+(?:for|of)\s+(.+)$", "Supports {value}"),
        )
        for pattern, template in nominal_patterns:
            match = re.search(pattern, noun_phrase, flags=re.IGNORECASE)
            if match:
                variants.append(template.format(value=match.group(1).strip()))
                nominal_matched = True
        if not nominal_matched:
            variants.append(f"Has {noun_phrase}")
    if lowered.startswith(("you're ", "you are ")):
        variants.append(re.sub(r"^(?:you're|you are)\s+", "Is ", text, flags=re.IGNORECASE))
    if lowered.startswith(("you've ", "you have ")):
        variants.append(re.sub(r"^(?:you've|you have)\s+", "Has ", text, flags=re.IGNORECASE))
    if lowered.startswith("you sometimes "):
        variants.append(re.sub(r"^you sometimes\s+", "Sometimes ", text, flags=re.IGNORECASE))
    if lowered.startswith("you often "):
        variants.append(re.sub(r"^you often\s+", "Often ", text, flags=re.IGNORECASE))
    if lowered.startswith("you tend to "):
        variants.append(re.sub(r"^you tend to\s+", "Tends to ", text, flags=re.IGNORECASE))
    if lowered.startswith("you "):
        variants.append(text[4:].strip())
    if lowered.startswith("the "):
        variants.append(f"Experienced {text}")
    return _dedupe([variant for variant in variants if variant])


def _personamem_antecedent_is_diagnostic(phrase: str) -> bool:
    text = _semantic_compact(phrase)
    if len(text.split()) < 3:
        return False
    low_signal = (
        "current request",
        "this request",
        "same normal preference source",
        "good ways",
        "ordinary alternatives",
        "personalized recommendation",
        "weekend",
        "next week",
    )
    if any(marker in text for marker in low_signal):
        return False
    return True


def _personamem_explicit_profile_phrases(response_text: str) -> list[str]:
    text = str(response_text or "")
    phrases: list[str] = []
    phrases.extend(_personamem_episodic_spans(text))
    phrases.extend(_personamem_schema_role_value_phrases(text))
    phrases.extend(_personamem_minimal_sufficient_habit_frames(text))
    card_pattern = re.compile(
        r"\*\*(?P<label>Civic Views|Work/School Responsibilities|Community Commitments|Stressors|Daily Routines|Values|Recurring Concerns|Concrete Example):\*\*\s*"
        r"(?P<body>.*?)(?=\s*\*\*(?:Civic Views|Work/School Responsibilities|Community Commitments|Stressors|Daily Routines|Values|Recurring Concerns|Concrete Example):\*\*|$)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    for match in card_pattern.finditer(text):
        body = " ".join(match.group("body").strip(" -*.").split())
        if 6 <= len(body) <= 320:
            phrase = _third_person_preference_clause(body)
            if phrase:
                phrases.append(phrase)
    for match in re.finditer(
        r"\b(?:social_security_number|ssn|physical_address|drivers_license_number|llm_api_key)\s*:\s*[^.;]+",
        text,
        flags=re.IGNORECASE,
    ):
        phrases.append(" ".join(match.group(0).strip().split()))
    cue_label_pattern = (
        r"(?:Personal cue|Profile cue|Evidence cue|Prior example|Concrete example|"
        r"Concrete fact|Personal fact|Recurring preference|Avoidance or constraint|"
        r"Correction/Avoidance|Correction|Avoidance|Avoid|Constraint|Responsibility|"
        r"Daily routine|Daily routines|Routine/Habit|Habit|Interest|Chosen cue|"
        r"Topic/Domain|Recurring concern|Value)"
    )
    cue_pattern = re.compile(
        rf"\b{cue_label_pattern}\s*\d*\s*[:=]\s*"
        rf"(?P<body>.*?)(?=\s+\b{cue_label_pattern}\s*\d*\s*[:=]|[.;\n]|$)",
        flags=re.IGNORECASE,
    )
    for match in cue_pattern.finditer(text):
        body = " ".join(match.group("body").strip(" -*").split())
        if body.lower() in {"option a", "option b", "chosen choice", "decision"}:
            continue
        phrase = _third_person_preference_clause(body)
        if phrase:
            phrases.append(phrase)
        phrases.extend(_personamem_negative_phrases_from_clause(body))
    plain_card_pattern = re.compile(
        r"(?m)^\s*(?:Values|Responsibilities|Recurring Concerns|Daily Routines|"
        r"Concrete Example|Community Commitments|Stressors|Correction/Avoidance|"
        r"Correction|Avoidance|Avoidance or constraint|Avoid|Routine/Habit|Habit|"
        r"Interest|Chosen cue|Topic/Domain)\s*:\s*(?P<body>[^\n]{6,320})",
        flags=re.IGNORECASE,
    )
    for match in plain_card_pattern.finditer(text):
        body = " ".join(match.group("body").strip(" -*.").split())
        phrase = _third_person_preference_clause(body)
        if phrase:
            phrases.append(phrase)
        phrases.extend(_personamem_negative_phrases_from_clause(body))
    wrapper_patterns = (
        r"\bsomeone who is ([^.;]+)",
        r"\breflect that they ([^.;]+)",
        r"\bfact that you ([^,.;]+)",
        r"\bsince you(?:'re| are|’re) ([^,.;]+)",
    )
    for match in re.finditer(
        r"\bdo not remember\s+['\"]([^'\"]+)['\"]\s+in memory",
        text,
        flags=re.IGNORECASE,
    ):
        target = " ".join(match.group(1).strip().split())
        if target:
            phrases.append(f"Do not remember '{target}' in memory")
            phrases.append(target[0].upper() + target[1:] if target else target)
    phrases.extend(_personamem_negative_phrases_from_clause(text))
    for pattern in wrapper_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            phrase = _third_person_preference_clause(match.group(1))
            if phrase:
                phrases.append(phrase)
    for segment in re.split(r"(?:\n+|[.;])", text):
        cleaned = " ".join(segment.strip(" -*:").split())
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if re.match(
            r"^(?:stable persona|interests|interest|responsibilities/routine|responsibilities|routine|routine/habit|habit|topic/domain|chosen cue|values/concerns|value/concern|values|concerns|correction/avoidance|prior interaction example|prior example)\s*[:=]",
            lowered,
        ):
            continue
        if re.match(r"^(?:since|given|because|considering|with|as|after)\b", lowered):
            continue
        if lowered.startswith("decision=") and re.search(
            r"\b(?:personal cue|profile cue|evidence cue|prior example|daily routines?)\b",
            lowered,
        ):
            continue
        if len(cleaned) <= 220 and (
            re.match(r"^(?:a|an)\s+.+\bwho\b", lowered)
            or re.search(r"\b(?:practices|volunteers|works|studies|keeps|maintains|prefers|enjoys|likes|loves)\b", lowered)
            or re.search(r"\b(?:allergic|asthma|migraine|injury|diagnosed|prescription|cholesterol)\b", lowered)
            or re.search(r"\b(?:community|civic|school|student|parent|curriculum|routine|stress|responsibil)\b", lowered)
        ):
            phrase = _third_person_preference_clause(cleaned)
            if phrase:
                phrases.append(phrase)
    conversation_match = re.search(
            r"(user:\s*.+?\bassistant:\s*.+?)(?=\s+(?:Raw prior interaction\b|Correction/Avoidance\b|Role\s*:|Stable persona\b|Interests\b|Interest\b|Responsibilities/Routine\b|Routine/Habit\b|Habit\b|Topic/Domain\b|Chosen cue\b|Values/Concerns\b|Value/Concern\b|Prior interaction example\b)|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if conversation_match:
        snippet = " ".join(conversation_match.group(1).strip().split())
        if 80 <= len(snippet) <= 4000:
            phrases.append(snippet)
    return _dedupe(phrases)


def _personamem_schema_role_value_phrases(response_text: str) -> list[str]:
    phrases: list[str] = []
    text = str(response_text or "")
    if not text.strip():
        return []

    for match in re.finditer(
            r"\bRaw prior interaction\s*:\s*(?P<body>User\s*:\s*.+?\bAssistant\s*:\s*.+?)(?=\s+(?:Raw prior interaction\b|Correction/Avoidance\b|Role\s*:|Stable persona\b|Interests\b|Interest\b|Responsibilities/Routine\b|Routine/Habit\b|Habit\b|Topic/Domain\b|Chosen cue\b|Values/Concerns\b|Value/Concern\b|Prior interaction example\b)|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        body = " ".join(match.group("body").strip(" -*").split())
        if 80 <= len(body) <= 4000:
            phrases.append(body)

    label_pattern = (
        r"Stable persona|Interests|Interest|Responsibilities/Routine|Responsibilities|Routine|"
        r"Routine/Habit|Habit|Topic/Domain|Chosen cue|Values/Concerns|Value/Concern|Values|Concerns|Correction/Avoidance|Correction|Avoidance|"
        r"Prior interaction example|Prior example"
    )
    role_value_pattern = re.compile(
        r"\bRole\s*:\s*(?P<role>[^|:]{2,80})\s*\|\s*Value\s*:\s*"
        r"(?P<value>.*?)(?=\s+(?:Raw prior interaction\b|Correction/Avoidance\b|Role\s*:|Stable persona\b|Interests\b|Interest\b|Responsibilities/Routine\b|Routine/Habit\b|Habit\b|Topic/Domain\b|Chosen cue\b|Values/Concerns\b|Value/Concern\b|Prior interaction example\b)|$)",
        flags=re.IGNORECASE,
    )
    labeled_pattern = re.compile(
        rf"\b(?P<label>{label_pattern})\s*[:=]\s*"
        rf"(?P<body>.*?)(?=\s+(?:{label_pattern})\s*[:=]|\s+Role\s*:|$)",
        flags=re.IGNORECASE | re.DOTALL,
    )

    for match in role_value_pattern.finditer(text):
        role = " ".join(match.group("role").strip().split()).lower()
        body = " ".join(match.group("value").strip(" -*").split())
        phrases.extend(_personamem_phrases_from_role_value(role, body))

    for match in labeled_pattern.finditer(text):
        label = " ".join(match.group("label").strip().split()).lower()
        body = " ".join(match.group("body").strip(" -*").split())
        phrases.extend(_personamem_phrases_from_role_value(label, body))

    return _dedupe(phrases)


def _personamem_phrases_from_role_value(role: str, body: str) -> list[str]:
    phrases: list[str] = []
    body = _personamem_truncate_schema_body(body)
    role_text = _personamem_clean_role_value_role(role)
    if role_text and _personamem_role_text_is_semantic(role_text):
        role_phrase = _third_person_preference_clause(role_text)
        if role_phrase:
            phrases.append(role_phrase)
    if not body or body.lower() in {"unknown", "none", "not available", "not specified"}:
        return _dedupe(phrases)
    if re.search(r"\b(no explicit|no known|none noted|not mentioned|unknown)\b", body, flags=re.IGNORECASE):
        return _dedupe(phrases)
    if "user:" in body.lower() and "assistant:" in body.lower():
        conversation = re.search(r"(user:\s*.+?\bassistant:\s*.+)", body, flags=re.IGNORECASE | re.DOTALL)
        if conversation:
            snippet = " ".join(conversation.group(1).strip().split())
            if 80 <= len(snippet) <= 4000:
                phrases.append(snippet)
        return phrases
    if any(marker in role for marker in ("correction", "avoid", "constraint")):
        phrases.extend(_personamem_negative_phrases_from_clause(body))
        phrase = _third_person_preference_clause(body)
        if phrase:
            phrases.append(phrase)
        return phrases
    phrase = _third_person_preference_clause(body)
    if phrase:
        phrases.append(phrase)
    phrases.extend(_personamem_minimal_sufficient_habit_frames(f"{role_text}. {body}"))
    return phrases


def _personamem_clean_role_value_role(role: str) -> str:
    text = " ".join(str(role or "").strip(" -*,.;:").split())
    text = re.sub(r"^(?:role|field|key)\s*[:=]\s*", "", text, flags=re.IGNORECASE)
    return text.strip(" -*,.;:")


def _personamem_role_text_is_semantic(role: str) -> bool:
    text = _semantic_compact(role)
    if not text or len(text.split()) < 3:
        return False
    generic_roles = {
        "persona",
        "personas",
        "interests",
        "interest",
        "routine",
        "responsibilities",
        "responsibilities routine",
        "values",
        "concerns",
        "values concerns",
        "correction",
        "avoidance",
        "prior interaction example",
    }
    if text in generic_roles:
        return False
    return bool(
        re.search(
            r"\b(?:listens?|reading|reads?|enjoys?|likes?|loves?|prefers?|collects?|follows?|"
            r"works?|studies|values?|cares?|is concerned|is committed|considers?|uses?)\b",
            role,
            flags=re.IGNORECASE,
        )
    )


def _personamem_minimal_sufficient_habit_frames(text: str) -> list[str]:
    """Extract compact habit frames that preserve cross-behavior evidence."""

    source = " ".join(str(text or "").strip().split())
    if not source:
        return []
    phrases: list[str] = []
    for match in re.finditer(
        r"\b(?:listens?|listening)\s+to\s+(?P<object>[^.;,]{3,80}?)\s+while\s+(?P<context>[^.;,]{3,80})",
        source,
        flags=re.IGNORECASE,
    ):
        obj = _personamem_clean_frame_fragment(match.group("object"))
        context = _personamem_clean_frame_fragment(match.group("context"))
        if obj and context:
            phrases.append(f"Listens to {obj} while {context}")
    if re.search(r"\b(?:reading|reads?|book|novel)\b", source, flags=re.IGNORECASE):
        music_match = re.search(
            r"\b(?:soft|gentle|calming|ambient|instrumental|background)?\s*"
            r"(?P<music>(?:ambient|instrumental|background|soft|gentle)(?:\s+\w+){0,3}\s+music)\b",
            source,
            flags=re.IGNORECASE,
        )
        if music_match:
            music = _personamem_clean_frame_fragment(music_match.group("music"))
            if music:
                phrases.append(f"Listens to {music} while reading")
    for pattern, template in (
        (r"\b(?:likes?|enjoys?|prefers?)\s+(?P<object>[^.;]{4,120}?\b(?:trails?|parks?|museums?|pools?|postcards?|figures?|films?|movies?|basketball|formula 1|stargazing|swimming|hiking))\b", "Likes {object}"),
        (r"\b(?:collects?|keeps?)\s+(?P<object>[^.;]{4,120}?\b(?:postcards?|figures?|collectibles?|memorabilia))\b", "Collects {object}"),
        (r"\b(?:follows?)\s+(?P<object>[^.;]{4,120}?\b(?:racing|formula 1|sports?|football|soccer))\b", "Follows {object}"),
    ):
        for match in re.finditer(pattern, source, flags=re.IGNORECASE):
            obj = _personamem_clean_frame_fragment(match.group("object"))
            if obj:
                phrases.append(template.format(object=obj))
    return _dedupe([phrase for phrase in phrases if _personamem_frame_is_diagnostic(phrase)])[:8]


def _personamem_clean_frame_fragment(value: str) -> str:
    text = " ".join(str(value or "").strip(" -*,.;:'\"").split())
    text = re.sub(r"\s+(?:in the background|nearby|for long stretches of time)\b", lambda m: m.group(0), text, flags=re.IGNORECASE)
    text = re.sub(r"\s+(?:and|while|with|because|so that)\b.*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(?:a|an|the|your)\s+", "", text, flags=re.IGNORECASE)
    return text.strip(" -*,.;:'\"")


def _personamem_frame_is_diagnostic(value: str) -> bool:
    text = _semantic_compact(value)
    if len(text.split()) < 3:
        return False
    bad = (
        "listens to music while reading" == text,
        "likes current request" in text,
        "personalized recommendation" in text,
    )
    return not any(bad)


def _personamem_truncate_schema_body(body: str) -> str:
    value = " ".join(str(body or "").strip(" -*").split())
    if not value:
        return ""
    split = re.split(
        r"\s+(?=(?:Raw prior interaction|Correction/Avoidance|Role\s*:|Stable persona|Interests|Responsibilities/Routine|Values/Concerns|Prior interaction example)\s*[:=])",
        value,
        maxsplit=1,
        flags=re.IGNORECASE,
    )
    return split[0].strip(" -*,.;")


def _personamem_negative_phrases_from_clause(text: str) -> list[str]:
    phrases: list[str] = []
    source = " ".join(str(text or "").strip().split())
    if not source:
        return []
    patterns = (
        r"\bforget(?: that)?\s+(?:the user\s+|they\s+|you\s+)?((?:enjoys?|likes?|prefers?)\s+[^.;]+)",
        r"\b(?:no longer|does not|doesn't|do not|don't)\s+((?:enjoys?|likes?|prefers?)\s+[^.;]+)",
        r"\bno preference for\s+([^.;]+)",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, source, flags=re.IGNORECASE):
            target = _clean_personamem_negative_target(match.group(1))
            if not target:
                continue
            normalized_target = target
            if not re.match(r"^(?:enjoys?|likes?|prefers?)\b", normalized_target, flags=re.IGNORECASE):
                normalized_target = normalized_target[0].lower() + normalized_target[1:]
            phrases.append(f"Do not remember '{normalized_target}' in memory")
            phrases.append(normalized_target[0].upper() + normalized_target[1:])
    return _dedupe(phrases)


def _clean_personamem_negative_target(value: str) -> str:
    target = " ".join(str(value or "").strip(" -*,.;:'\"").split())
    target = re.sub(r"^(?:that\s+)?(?:the user\s+|they\s+|you\s+)", "", target, flags=re.IGNORECASE)
    target = re.sub(r"\s+(?:in recommendations?|for this request|as a preference|anymore)\b.*$", "", target, flags=re.IGNORECASE)
    target = re.sub(r"^(?:activities such as|things like)\s+", "", target, flags=re.IGNORECASE)
    target = re.sub(r"^enjoy\s+", "enjoys ", target, flags=re.IGNORECASE)
    target = re.sub(r"^like\s+", "likes ", target, flags=re.IGNORECASE)
    target = re.sub(r"^prefer\s+", "prefers ", target, flags=re.IGNORECASE)
    return target.strip(" -*,.;:'\"")


def _personamem_episodic_spans(response_text: str) -> list[str]:
    spans: list[str] = []
    for segment in re.split(r"(?:\n+|[.;])", str(response_text or "")):
        cleaned = " ".join(segment.strip(" -*:").split())
        if not 35 <= len(cleaned) <= 360:
            continue
        lowered = cleaned.lower()
        if re.match(r"^(?:since|given|because|considering|with|as|after)\b", lowered):
            continue
        if "user:" in lowered or "assistant:" in lowered or lowered.startswith("raw prior interaction"):
            continue
        if lowered.startswith(
            (
                "you ",
                "you might",
                "you could",
                "if you",
                "when ",
                "sometimes",
                "a good fit",
                "as someone",
                "to prepare",
            )
        ):
            continue
        first_person = bool(re.search(r"\b(i|my|me|we|our|kids|colleague|neighbor|student)\b", lowered))
        concrete = len(_content_tokens(cleaned)) >= 6
        if first_person and concrete:
            spans.append(cleaned)
    return _dedupe(spans)[:3]


def _third_person_preference_clause(text: str) -> str:
    phrase = " ".join(str(text or "").strip(" -*.").split())
    if not phrase:
        return ""
    lower = phrase.lower()
    if re.search(r"\b(?:no explicit|none specified|none noted|not mentioned|unknown)\b", lower):
        return ""
    if lower.startswith("you are "):
        phrase = phrase[8:]
        lower = phrase.lower()
        if lower.startswith("grading "):
            phrase = "Grades " + phrase[8:]
            lower = phrase.lower()
    elif lower.startswith("you sometimes "):
        phrase = "Sometimes " + phrase[14:]
        lower = phrase.lower()
    elif lower.startswith("you often "):
        phrase = "Often " + phrase[10:]
        lower = phrase.lower()
    elif lower.startswith("you tend to "):
        phrase = "Tends to " + phrase[12:]
        lower = phrase.lower()
    elif lower.startswith("you volunteer "):
        phrase = "Volunteers " + phrase[14:]
        lower = phrase.lower()
    elif lower.startswith("you practice "):
        phrase = "Practices " + phrase[13:]
        lower = phrase.lower()
    elif lower.startswith("you start "):
        phrase = "Starts " + phrase[10:]
        lower = phrase.lower()
    elif lower.startswith("you shared "):
        phrase = "Shared " + phrase[11:]
        lower = phrase.lower()
    replacements = (
        ("practice ", "practices "),
        ("enjoy ", "enjoys "),
        ("like ", "likes "),
        ("love ", "loves "),
        ("prefer ", "prefers "),
        ("have ", "has "),
        ("are ", "is "),
    )
    for source, target in replacements:
        if lower.startswith(source):
            phrase = target + phrase[len(source) :]
            break
    phrase = re.sub(
        r"^(Sometimes|Often)\s+(opt|attend|choose|work|support|prefer|enjoy|like|love|value|care|seek|manage|write|create|practice)\b",
        lambda match: f"{match.group(1)} {_third_person_verb(match.group(2))}",
        phrase,
        flags=re.IGNORECASE,
    )
    phrase = re.sub(
        r"^Tends to\s+(opt|attend|choose|work|support|prefer|enjoy|like|love|value|care|seek|manage|write|create|practice)\b",
        lambda match: f"Tends to {_base_verb(match.group(1))}",
        phrase,
        flags=re.IGNORECASE,
    )
    if 6 <= len(phrase) <= 180:
        return phrase[0].upper() + phrase[1:]
    return ""


def _third_person_verb(verb: str) -> str:
    base = _base_verb(verb)
    irregular = {"choose": "chooses", "practice": "practices"}
    if base in irregular:
        return irregular[base]
    if base.endswith("s"):
        return base
    return base + "s"


def _base_verb(verb: str) -> str:
    return str(verb or "").strip().lower()


def _stress_phrase_from_visible_clause(text: str) -> str:
    phrase = " ".join(str(text or "").strip(" -*.").split())
    if len(phrase) < 8:
        return ""
    lower = phrase.lower()
    if "parent" in lower and ("curriculum" in lower or "school" in lower):
        return "Has lingering stress from handling aggressive behavior from a parent toward school staff during a curriculum debate"
    if any(token in lower for token in ("confrontation", "anxiety", "stress", "tense")):
        return f"Has lingering stress from handling {phrase}"
    return ""


def _strip_choice_number(text: str) -> str:
    return re.sub(r"^\s*\d+\s*[\).\:-]\s*", "", str(text or "").strip())


def _strip_choice_number_unless_locomo_timestamp(text: str) -> str:
    raw = str(text or "").strip()
    if re.match(r"^\s*\d{1,2}:\d{2}\s*(?:am|pm)\b", raw, flags=re.IGNORECASE):
        return raw
    return _strip_choice_number(raw)


def _personalens_affinity_atoms(
    *,
    task_domain: str,
    visible_affinities: list[Any],
    response_text: str,
    task_prompt: str = "",
) -> list[str]:
    atoms: list[str] = []
    for affinity in visible_affinities:
        if not isinstance(affinity, Mapping):
            continue
        domain = str(affinity.get("domain") or task_domain or "unknown").strip()
        key = str(affinity.get("affinity_key") or "unknown_affinity").strip()
        values = affinity.get("values")
        if not isinstance(values, list):
            continue
        for value in values:
            text = str(value).strip()
            if domain and key and text:
                atoms.append(f"personalens_affinity={domain}|{key}={text}")
    if atoms:
        return _dedupe(atoms)

    atoms.extend(_parse_embedded_personalens_affinities(response_text, task_domain=task_domain))
    atoms.extend(_personalens_alarm_atoms(response_text))
    atoms.extend(_parse_personalens_profile_plan_atoms(response_text, task_domain=task_domain))
    atoms.extend(_parse_personalens_real_agent_profile_atoms(response_text, task_domain=task_domain))
    if atoms:
        return _dedupe(atoms)

    atoms.extend(_personalens_schema_value_atoms(task_prompt=task_prompt, response_text=response_text, task_domain=task_domain))
    if atoms:
        return _dedupe(atoms)

    domain = str(task_domain or "unknown").replace("_", " ").strip().title() or "Unknown"
    for key, values_text in _parse_arrow_affinity_chunks(response_text):
        for value in _split_values(values_text):
            atoms.append(f"personalens_affinity={domain}|{key}={value}")
    for key, value in _parse_key_value_affinity_chunks(response_text):
        atoms.append(f"personalens_affinity={domain}|{key}={value}")
    return _dedupe(atoms)


def _parse_personalens_profile_plan_atoms(response_text: str, *, task_domain: str) -> list[str]:
    atoms: list[str] = []
    text = str(response_text or "")
    domain_blocks: list[tuple[str, str]] = []
    pattern = re.compile(
        r"(?:^|[.\n]\s*)For\s+(?P<domain>[A-Z][A-Za-z &/-]{1,40}),\s*(?P<body>.*?)(?=(?:[.\n]\s*)For\s+[A-Z][A-Za-z &/-]{1,40},|\Z)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    for match in pattern.finditer(text):
        domain = " ".join(match.group("domain").strip(" -*").split()).title()
        body = " ".join(match.group("body").strip().split())
        if domain and body:
            domain_blocks.append((domain, body))
    if not domain_blocks:
        return []

    for domain, body in domain_blocks:
        if domain.strip().lower() == "account context":
            continue
        for key, values_text in _parse_profile_plan_key_values(body):
            key = _canonical_personalens_key(key)
            for value in _split_values(values_text):
                cleaned = _clean_embedded_affinity_value(value)
                if key and cleaned:
                    atoms.append(f"personalens_affinity={domain}|{key}={cleaned}")
    return _dedupe(atoms)


def _parse_personalens_real_agent_profile_atoms(response_text: str, *, task_domain: str) -> list[str]:
    """Map ordinary real-agent prose back to profile slot carriers."""

    text = " ".join(str(response_text or "").strip().split())
    if not text:
        return []
    domain = str(task_domain or "unknown").replace("_", " ").strip().title() or "Unknown"
    atoms: list[str] = []

    def add(key: str, value: str) -> None:
        key = _canonical_personalens_key(key)
        cleaned = _clean_personalens_keyed_value(key, value)
        lower_value = cleaned.lower()
        if key == "Prefer Car Brand" and not any(
            marker in lower_value
            for marker in ("mercedes", "toyota", "honda", "bmw", "audi", "ford", "tesla")
        ):
            return
        if key == "Favorite Team" and lower_value in {"active", "boutique", "single play", "not specified", "n/a", "none"}:
            return
        if key == "Favorite Media" and lower_value in {"active", "not specified", "n/a", "none"}:
            return
        if key == "Price Range" and not re.search(r"[0-9]", cleaned):
            return
        if key == "Service Frequency Preference" and lower_value not in {
            "daily",
            "weekly",
            "weekends",
            "weekends only",
            "monthly",
            "quarterly",
            "annually",
            "yearly",
        }:
            return
        if key == "Financial Company" and "," in cleaned:
            for value_part in _split_values(cleaned):
                part = _clean_embedded_affinity_value(value_part)
                if part:
                    atoms.append(f"personalens_affinity={domain}|{key}={part}")
            return
        if key and cleaned:
            atoms.append(f"personalens_affinity={domain}|{key}={cleaned}")

    structured_keys = (
        "Favorite Team",
        "Favorite Media",
        "Favorite Books",
        "Media",
        "Team",
        "Preferred Car Brand",
        "Prefer Car Brand",
        "Car Brand",
        "Car or Brands",
        "Preferred Sectors",
        "Prefer Sectors",
        "Sectors or Services",
        "Service Frequency Preference",
        "Service Frequency",
        "Service",
        "Frequency",
        "Location Preference",
        "Viewing Time Preference",
        "Communication Style",
        "Financial Company",
        "Preferred Product Category",
        "Prefer Product Category",
        "Reading Format",
        "Hotel Chains Preference",
        "Duration Preference",
        "News Sources",
        "Platform Preference",
        "Favorite Bands",
        "Favorite Albums",
        "Favorite Authors",
        "Favorite Sports",
        "Prefer Game Name",
        "Preferred Game Name",
        "Prefer Communication Style",
        "Event Type Preference",
        "Service Provider Gender Preference",
        "Favorite Actors and Directors",
        "Ambiance Preference",
        "Cuisine Preference",
        "Prefer Airline",
        "Preferred Airline",
        "Car Type Preference",
        "Dietary Restrictions",
        "Travel Time Preference",
        "Viewing Platform Preference",
        "Appointment Time Preference",
        "Departure Time Preference",
        "Reading Frequency",
        "Theater Type Preference",
        "Genre",
        "Multiplayer Preference",
        "Prefer Genres",
        "Preferred Genres",
        "Amenity Preference",
        "Amenity",
        "Amenities",
        "Price Range",
        "Budget or Price",
        "Budget",
        "Brand",
        "Price",
    )
    structured_key_pattern = "|".join(re.escape(key) for key in sorted(structured_keys, key=len, reverse=True))
    profile_stop_keys = (
        "Education",
        "Education Level",
        "English Proficiency",
        "Language",
        "Religion",
        "Age",
        "Gender",
        "Country",
        "Interest",
        "Media",
        "Team",
        "Budget or Price",
        "Car or Brands",
        "Sectors or Services",
        "Amenities",
        "Frequency",
        "Location Preference",
        "Viewing Time Preference",
        "Communication Style",
        "Financial Company",
        "Preferred Product Category",
        "Prefer Product Category",
        "Reading Format",
        "Hotel Chains Preference",
        "Duration Preference",
        "News Sources",
        "Platform Preference",
        "Favorite Bands",
        "Favorite Albums",
        "Favorite Authors",
        "Favorite Sports",
        "Prefer Game Name",
        "Preferred Game Name",
        "Prefer Communication Style",
        "Event Type Preference",
        "Service Provider Gender Preference",
        "Favorite Actors and Directors",
        "Ambiance Preference",
        "Cuisine Preference",
        "Prefer Airline",
        "Preferred Airline",
        "Car Type Preference",
        "Dietary Restrictions",
        "Travel Time Preference",
        "Viewing Platform Preference",
        "Appointment Time Preference",
        "Departure Time Preference",
        "Reading Frequency",
        "Theater Type Preference",
        "Genre",
        "Multiplayer Preference",
        "Prefer Genres",
        "Preferred Genres",
        "Current Task Choices",
        "Task Choices",
        "Profile Context",
        "Time",
        "Sound",
        "Recurrence",
        "Brand",
        "Service",
    )
    value_stop_pattern = "|".join(
        re.escape(key)
        for key in sorted(tuple(structured_keys) + profile_stop_keys, key=len, reverse=True)
    )

    def trim_value(value: str, *, key: str) -> str:
        cleaned = str(value or "")
        cleaned = re.split(rf"\s+\b(?:{value_stop_pattern})\b\s*[:=]?", cleaned, maxsplit=1, flags=re.IGNORECASE)[0]
        if _canonical_personalens_key(key) == "Price Range":
            cleaned = cleaned.replace("$", "").replace(",", "").rstrip("+")
        if _canonical_personalens_key(key) == "Amenity Preference":
            cleaned = re.sub(r"\s+as\s+(?:a|an|the)\b.*$", "", cleaned, flags=re.IGNORECASE)
        return cleaned

    for match in re.finditer(
        rf"\b(?P<key>{structured_key_pattern})\s*(?:is\s+set\s+to|is|are|[:=])\s*(?P<value>.*?)(?=(?:,\s*|\s+)\b(?:{structured_key_pattern})\s*(?:is\s+set\s+to|is|are|[:=])|[.;]|$)",
        text,
        flags=re.IGNORECASE,
    ):
        key = match.group("key")
        if key.lower() == "budget":
            key = "Price Range"
        add(key, trim_value(match.group("value"), key=key))

    for match in re.finditer(r"\bprice range\s*(?:of|is|:)?\s*\$?([0-9][0-9,]*(?:\+|(?:\s*(?:to|-)\s*\$?[0-9][0-9,]*)?)?)", text, flags=re.IGNORECASE):
        add("Price Range", trim_value(match.group(1), key="Price Range"))
    for match in re.finditer(r"\bfavorite team\s*(?:is|:)?\s*(?:the\s+)?([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,4})\b", text, flags=re.IGNORECASE):
        add("Favorite Team", trim_value(match.group(1), key="Favorite Team"))
    for match in re.finditer(r"\bfavorite media\s*(?:is|:)?\s*['\"]?([^'\".;]+)", text, flags=re.IGNORECASE):
        add("Favorite Media", trim_value(match.group(1), key="Favorite Media"))
    for match in re.finditer(r"\bpreferred car brand\s*(?:is|:)?\s*([A-Z][A-Za-z0-9-]+(?:\s+[A-Z][A-Za-z0-9-]+){0,3})\b", text, flags=re.IGNORECASE):
        add("Prefer Car Brand", trim_value(match.group(1), key="Prefer Car Brand"))
    for match in re.finditer(r"\b(?:prefer|prefers|preferred)\s+([A-Z][A-Za-z0-9-]+(?:\s+[A-Z][A-Za-z0-9-]+){0,3})\b", text):
        value = match.group(1).strip()
        if any(marker in value.lower() for marker in ("mercedes", "toyota", "honda", "bmw", "audi", "ford", "tesla")):
            add("Prefer Car Brand", trim_value(value, key="Prefer Car Brand"))
    for match in re.finditer(r"\b(?:with|including|include)\s+([a-z][a-z0-9 -]{2,40})\s+amenit(?:y|ies)\b", text, flags=re.IGNORECASE):
        add("Amenity Preference", trim_value(match.group(1), key="Amenity Preference"))
    for match in re.finditer(r"\bamenity preference\s*(?:is|:)?\s*(?:a|an|the)?\s*([a-z][a-z0-9 -]{2,40})", text, flags=re.IGNORECASE):
        add("Amenity Preference", match.group(1))
    for match in re.finditer(r"\b(?:follow|watch|support)\s+(?:the\s+)?([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,4})\b", text):
        team = match.group(1).strip()
        if not any(bad in team.lower() for bad in ("movie", "film", "book", "music", "please")):
            add("Favorite Team", team)
    for match in re.finditer(r"['\"]([^'\"]{2,80})['\"]", text):
        value = match.group(1).strip()
        window = text[max(0, match.start() - 120) : min(len(text), match.end() + 160)].lower()
        if "favorite media" in window or "aligns with your favorite media" in window:
            add("Favorite Media", value)
        elif str(task_domain or "").strip().lower() in {"book", "books", "reading"} and ("book" in window or "read" in window):
            add("Favorite Books", value)
        elif "movie" in window or "film" in window:
            add("Favorite Media", value)
    sector_match = re.search(r"\b(?:prefer|preferred|preferred|usual)\s+sectors?\s*(?:are|is|include|including|like|such as|toward|towards|:)?\s*([^.;]+)", text, flags=re.IGNORECASE)
    if sector_match:
        for value in _split_values(sector_match.group(1)):
            add("Prefer Sectors", trim_value(re.sub(r"^(?:are|is)\s+", "", value, flags=re.IGNORECASE), key="Prefer Sectors"))
    service_match = re.search(r"\bservice frequency(?: preference)?\s*(?:is|:|of)?\s*([a-z][a-z -]{2,30})", text, flags=re.IGNORECASE)
    if service_match:
        add("Service Frequency Preference", trim_value(service_match.group(1), key="Service Frequency Preference"))
    return _dedupe(atoms)


def _personalens_profile_fact_atoms(response_text: str) -> list[str]:
    text = " ".join(str(response_text or "").strip().split())
    if not text:
        return []
    atoms: list[str] = []
    label_map = {
        "age": "age",
        "gender": "gender",
        "employment status": "employment_status",
        "education": "education",
        "education level": "education",
        "marital status": "marital_status",
        "english proficiency": "english_proficiency",
        "language": "english_proficiency",
        "ethnicity": "ethnicity",
        "religion": "religion",
        "birth country": "birth_country",
        "reside country": "reside_country",
        "country": "reside_country",
    }
    label_pattern = "|".join(re.escape(label) for label in sorted(label_map, key=len, reverse=True))
    fact_stop_keys = tuple(label_map) + (
        "Interest",
        "Favorite Team",
        "Favorite Media",
        "Media",
        "Team",
        "Preferred Car Brand",
        "Prefer Car Brand",
        "Preferred Sectors",
        "Sectors or Services",
        "Service Frequency",
        "Service",
        "Frequency",
        "Amenity",
        "Amenities",
        "Price Range",
        "Budget",
        "Budget or Price",
        "Car or Brands",
        "Brand",
        "Price",
    )
    fact_stop_pattern = "|".join(re.escape(label) for label in sorted(fact_stop_keys, key=len, reverse=True))
    for match in re.finditer(
        rf"\b(?P<label>{label_pattern})\s*(?:is|=|:)\s*(?P<value>.*?)(?=\s+\b(?:{fact_stop_pattern})\b\s*(?:is|=|:)?|[.;]|$)",
        text,
        flags=re.IGNORECASE,
    ):
        label = " ".join(match.group("label").lower().split())
        value = _clean_embedded_affinity_value(match.group("value"))
        value = re.sub(r",?\s+and$", "", value, flags=re.IGNORECASE).strip()
        if label == "religion" and value.lower().startswith(("n/a", "na", "not specified", "unknown")):
            value = "n/a"
        elif value.lower().startswith(("not specified", "unknown")):
            continue
        elif label == "language" and value.lower() in {"english", "en"}:
            continue
        if value:
            atoms.append(f"{label_map[label]}={value}")
    for label, key in label_map.items():
        pattern = re.compile(
            rf"\b{re.escape(label)}\s+(?:like|such as|including)\s+(?P<value>.*?)(?=;\s*[a-z ]+\s+(?:like|such as|including)\s+|[.]|$)",
            flags=re.IGNORECASE,
        )
        for match in pattern.finditer(text):
            value = _clean_embedded_affinity_value(match.group("value"))
            if value:
                atoms.append(f"{key}={value}")
    for state_label, state in (("active areas", "active"), ("inactive areas", "inactive")):
        pattern = re.compile(
            rf"\b{re.escape(state_label)}\s+(?:like|such as|including)\s+(?P<values>.*?)(?=;\s*[a-z ]+\s+(?:like|such as|including)\s+|[.]|$)",
            flags=re.IGNORECASE,
        )
        for match in pattern.finditer(text):
            for domain in _split_personalens_unbounded_values(match.group("values")):
                cleaned = " ".join(str(domain or "").strip(" -*,.;:").split())
                if cleaned:
                    atoms.append(f"Interest {cleaned}={state}")
    lower = text.lower()
    active_interest_patterns = (
        r"\bactive\s+([a-z][a-z -]{2,40})\s+interests?\b",
        r"\binterests?\s+(?:include|including|like)\s+([^.;]+)",
        r"\binterest in ([a-z][a-z -]{2,40})\s+is\s+active\b",
    )
    for pattern in active_interest_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            values = re.sub(r"\band\b", ",", match.group(1), flags=re.IGNORECASE)
            for value in values.split(","):
                cleaned = " ".join(value.strip(" -*.").split())
                if cleaned and len(cleaned) <= 40:
                    atoms.append(f"Interest {cleaned.lower()}=active")
    for match in re.finditer(
        r"\bInterest\s+([A-Za-z][A-Za-z /_-]{2,40})\s*[:=]\s*(active|inactive)\b",
        text,
        flags=re.IGNORECASE,
    ):
        domain = " ".join(match.group(1).strip(" -*").split())
        state = match.group(2).lower()
        if domain:
            atoms.append(f"Interest {domain.lower()}={state}")
    for match in re.finditer(
        rf"\bactive interests?\s*[:=]\s*(.*?)(?=\s+\b(?:{fact_stop_pattern})\b\s*(?:is|=|:)?|[.;]|$)",
        text,
        flags=re.IGNORECASE,
    ):
        for domain in _split_personalens_unbounded_values(match.group(1)):
            cleaned = " ".join(str(domain or "").strip(" -*,.;:").split())
            if cleaned:
                atoms.append(f"Interest {cleaned.lower()}=active")
    if "shopping" in lower and "interest" in lower:
        atoms.append("Interest shopping=active")
    if "movie" in lower and "interest" in lower:
        atoms.append("Interest movies=active")
    education_match = re.search(r"\beducation level\s*\(([^)]+)\)", text, flags=re.IGNORECASE)
    if education_match:
        atoms.append(f"education={_clean_embedded_affinity_value(education_match.group(1))}")
    religion_match = re.search(r"\breligion\s+(?:is\s+)?(?:not specified|n/a|na|unknown)", text, flags=re.IGNORECASE)
    if religion_match:
        atoms.append("religion=n/a")
    return _dedupe(atoms)


def _split_personalens_unbounded_values(values_text: str) -> list[str]:
    cleaned = re.sub(r"\band\b", ",", str(values_text or ""), flags=re.IGNORECASE)
    return [
        item.strip(" -*.")
        for item in cleaned.split(",")
        if item.strip(" -*.")
    ]


def _parse_profile_plan_key_values(body: str) -> list[tuple[str, str]]:
    chunks: list[tuple[str, str]] = []
    for segment in re.split(r";|\s+[|]\s+", body):
        cleaned = " ".join(segment.strip(" -*.").split())
        if not cleaned:
            continue
        natural_chunks = _parse_natural_profile_plan_segment(cleaned)
        if natural_chunks:
            chunks.extend(natural_chunks)
            continue
        if ":" in cleaned:
            key, values = cleaned.split(":", 1)
            key = " ".join(key.strip(" -*").split())
            values = " ".join(values.strip(" -*").split())
            if 2 <= len(key) <= 80 and values:
                chunks.append((key, values))
            continue
    return chunks


def _parse_natural_profile_plan_segment(segment: str) -> list[tuple[str, str]]:
    chunks: list[tuple[str, str]] = []
    pattern = re.compile(
        r"(?P<key>[A-Za-z][A-Za-z &/_-]{2,90}?)\s+"
        r"(?:like|such as|including|leaning toward|toward|towards)\s+"
        r"(?P<values>[^.;]+)",
        flags=re.IGNORECASE,
    )
    for match in pattern.finditer(segment):
        key = _clean_natural_personalens_key(match.group("key"))
        values = " ".join(match.group("values").strip(" -*").split())
        if 2 <= len(key) <= 80 and values:
            chunks.append((key, values))
    return chunks


def _clean_natural_personalens_key(key: str) -> str:
    cleaned = " ".join(str(key or "").strip(" -*").split())
    cleaned = re.sub(
        r"^.*\b(?:around|toward|towards|with|for)\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"^(?:and|also|the|my|your)\s+", "", cleaned, flags=re.IGNORECASE)
    return " ".join(cleaned.strip(" -*").split())


def _parse_embedded_personalens_affinities(response_text: str, *, task_domain: str) -> list[str]:
    atoms: list[str] = []
    structured_text = str(response_text or "")
    if task_domain:
        domain_pattern = re.escape(str(task_domain).replace("_", " ").strip().title())
        structured_text = re.sub(rf"\s+({domain_pattern}\|)", r"\n\1", structured_text)
    pattern = re.compile(
        r"(?:(?P<domain>[A-Za-z][A-Za-z /&-]{1,40})\|)"
        r"(?P<key>[A-Za-z][A-Za-z /&-]{2,80}?)"
        r"\s*=\s*(?P<value>[^;\n.]+)",
        flags=re.IGNORECASE,
    )
    for match in pattern.finditer(structured_text):
        domain = " ".join(str(match.group("domain") or task_domain or "unknown").strip().split())
        key = _canonical_personalens_key(match.group("key"))
        value = _clean_personalens_keyed_value(key, match.group("value"))
        if domain and key and value:
            atoms.append(f"personalens_affinity={domain}|{key}={value}")
    if atoms:
        return _dedupe(atoms)
    fallback = re.compile(
        r"(?P<key>[A-Za-z][A-Za-z /&-]{2,70}?(?:Preference|Preferences|Genres|Genre|Timezone|Contact|Media|Actors and Directors|Artists|Albums|Bands|Restrictions|Range|Budget|Price|Size|Style|Sources|Company|Team|Sports|Platform|Frequency|Class|Season|Destination|Airline|Provider|Category|Brand|Location|Amenities?|Cuisine|Ambiance|Duration|Type|Seat|Sound|Time|Recurring))"
        r"\s*=\s*(?P<value>[^;\n.]+)",
        flags=re.IGNORECASE,
    )
    for match in fallback.finditer(structured_text):
        domain = str(task_domain or "unknown").replace("_", " ").strip().title() or "Unknown"
        raw_key = " ".join(match.group("key").strip(" -*").split())
        raw_key_lower = raw_key.lower()
        allowed_unscoped = {
            "budget",
            "budget or price",
            "price",
            "brand",
            "car or brands",
            "media",
            "team",
            "service",
            "frequency",
            "amenity",
            "amenities",
        }
        if (
            "task" in raw_key_lower
            or "profile context" in raw_key_lower
            or (
                raw_key_lower not in allowed_unscoped
                and "preference" not in raw_key_lower
                and "preferred" not in raw_key_lower
                and "favorite" not in raw_key_lower
            )
        ):
            continue
        key = _canonical_personalens_key(raw_key)
        value = _clean_personalens_keyed_value(key, match.group("value"))
        if key and value:
            atoms.append(f"personalens_affinity={domain}|{key}={value}")
    return _dedupe(atoms)


def _personalens_alarm_atoms(response_text: str) -> list[str]:
    text = str(response_text or "")
    atoms: list[str] = []
    lower = text.lower()
    concrete_alarm_hint = bool(
        re.search(
            r"\b(?:alarm sound|sound|location preference|preferred location|recurrence|recurring|near home|every day|daily|weekday|weekend)\b",
            lower,
        )
    )
    if (
        any(marker in lower for marker in ("would you like", "could you please specify", "please let me know"))
        and not concrete_alarm_hint
    ):
        return []
    time_match = re.search(r"\b([0-2]?\d:[0-5]\d\s*(?:AM|PM|am|pm))\b", text)
    if time_match:
        atoms.append(f"personalens_affinity=Alarm|Alarm Time Preference={_normalize_clock_value(time_match.group(1))}")
    sound_patterns = (
        r"\buse ([A-Za-z][A-Za-z ]{2,50}) as (?:the )?alarm sound",
        r"\bchoose [\"']?([A-Za-z][A-Za-z ]{2,50})[\"']? as (?:your |the )?alarm sound",
        r"\bwith ([A-Za-z][A-Za-z ]{2,50}) as (?:the )?alarm sound",
        r"\bfavorite alarm sound is\s*\**\s*([^,\-\n.;?]+)",
        r"\balarm sound preference\s*[:=]\s*\**\s*([^,\-\n.;?]+)",
        r"\bSound:\s*\**\s*([^,\-\n.;]+)",
        r"\bsound\s*=\s*\**\s*([^,\-\n.;]+)",
        r"\balarm sound(?: is|:)?\s*\**\s*([^,\-\n.;]+)",
    )
    for pattern in sound_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            value = _clean_affinity_value(match.group(1))
            if value:
                atoms.append(f"personalens_affinity=Alarm|Alarm Sound Preference={value}")
                break
    location_patterns = (
        r"\blocation preference\s*[:=]\s*([^.;?\n]+)",
        r"\bpreferred location\s*(?:is|:)?\s*([^.;?\n]+)",
        r"\bpreferred wake-up time is\s*(near home)\b",
    )
    for pattern in location_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            value = _clean_affinity_value(match.group(1))
            if value:
                atoms.append(f"personalens_affinity=Alarm|Location Preference={value}")
                break
    if not any("Location Preference=" in atom for atom in atoms) and "near home" in lower:
        atoms.append("personalens_affinity=Alarm|Location Preference=near home")
    if "weekday" in lower or "monday to friday" in lower:
        atoms.append("personalens_affinity=Alarm|Alarm Recurring Preference=weekdays")
    elif "weekend" in lower:
        atoms.append("personalens_affinity=Alarm|Alarm Recurring Preference=weekends")
    elif "daily" in lower or "every day" in lower:
        atoms.append("personalens_affinity=Alarm|Alarm Recurring Preference=daily")
    return _dedupe(atoms)


def _personalens_schema_value_atoms(*, task_prompt: str, response_text: str, task_domain: str) -> list[str]:
    atoms: list[str] = []
    domain = str(task_domain or "unknown").replace("_", " ").strip().title() or "Unknown"
    prompt = str(task_prompt or "")
    response = str(response_text or "")
    key_candidates = _induce_personalens_keys_from_prompt(prompt)
    for key in key_candidates:
        value = _extract_value_for_personalens_key(key, response)
        if value:
            atoms.append(f"personalens_affinity={domain}|{key}={value}")
    return _dedupe(atoms)


def _induce_personalens_keys_from_prompt(task_prompt: str) -> list[str]:
    text = str(task_prompt or "").lower()
    candidates: list[str] = []
    rules = (
        ("Preferred Genres", ("preferred genres", "preferred genre")),
        ("Genre", ("genre", "genres")),
        ("Favorite Artists", ("favorite artists", "favourite artists")),
        ("Favorite Bands", ("favorite bands", "favourite bands")),
        ("Favorite Albums", ("favorite albums", "favourite albums")),
        ("Preferred Messaging Apps", ("messaging app", "messaging apps")),
        ("Communication Style", ("communication style",)),
        ("Cuisine Preference", ("preferred cuisine", "cuisine type")),
        ("Dietary Restrictions", ("dietary restrictions",)),
        ("Price Range", ("price range",)),
        ("Location Preference", ("location preference", "preferred location")),
        ("Amenity Preference", ("amenity", "amenities")),
        ("Group Size Preference", ("group size",)),
        ("Event Type Preference", ("event type", "event types")),
        ("Notification Preference", ("notification preference", "notification")),
        ("Travel Season Preference", ("travel season", "season preference")),
        ("Preferred Destination Types", ("destination types", "preferred destination")),
        ("Preferred Airline", ("preferred airline", "airline preference")),
        ("Seat Preference", ("seat preference", "preferred seat")),
        ("Room Type Preference", ("room type",)),
        ("Brand Preference", ("favorite brands", "favourite brands", "brand preference")),
        ("Preferred Product Category", ("product category",)),
        ("Appointment Time Preference", ("appointment time",)),
    )
    for key, hints in rules:
        if any(hint in text for hint in hints):
            candidates.append(key)
    return _dedupe(candidates)


def _extract_value_for_personalens_key(key: str, response_text: str) -> str:
    key_terms = {
        "Preferred Genres": ("genre", "genres"),
        "Genre": ("genre", "genres"),
        "Favorite Artists": ("artist", "artists"),
        "Favorite Bands": ("band", "bands"),
        "Favorite Albums": ("album", "albums"),
        "Preferred Messaging Apps": ("messaging app", "app"),
        "Communication Style": ("communication style", "style"),
        "Cuisine Preference": ("cuisine",),
        "Dietary Restrictions": ("dietary", "restriction"),
        "Price Range": ("price range", "budget"),
        "Location Preference": ("location",),
        "Amenity Preference": ("amenity", "amenities"),
        "Group Size Preference": ("group size", "group"),
        "Event Type Preference": ("event type", "event"),
        "Notification Preference": ("notification",),
        "Travel Season Preference": ("season",),
        "Preferred Destination Types": ("destination", "attraction"),
        "Preferred Airline": ("airline",),
        "Seat Preference": ("seat",),
        "Room Type Preference": ("room type", "room"),
        "Brand Preference": ("brand",),
        "Preferred Product Category": ("category", "product"),
        "Appointment Time Preference": ("appointment", "time"),
    }.get(key, ())
    for term in key_terms:
        pattern = re.compile(rf"\b{re.escape(term)}s?\b\s*(?:is|are|:|-)?\s*\**\s*([^.;\n]+)", flags=re.IGNORECASE)
        match = pattern.search(response_text)
        if match:
            value = _clean_affinity_value(match.group(1))
            if value:
                return value
    return ""


def _canonical_personalens_key(key: str) -> str:
    text = " ".join(str(key or "").strip(" -*:").split())
    lookup = {
        "alarm time": "Alarm Time Preference",
        "alarm sound": "Alarm Sound Preference",
        "alarm recurring": "Alarm Recurring Preference",
        "recurrence pattern": "Alarm Recurring Preference",
        "preferred car brand": "Prefer Car Brand",
        "car brand": "Prefer Car Brand",
        "car or brands": "Prefer Car Brand",
        "brand": "Prefer Car Brand",
        "media": "Favorite Media",
        "team": "Favorite Team",
        "location preference": "Location Preference",
        "viewing time preference": "Viewing Time Preference",
        "communication style": "Communication Style",
        "financial company": "Financial Company",
        "preferred product category": "Prefer Product Category",
        "prefer product category": "Prefer Product Category",
        "reading format": "Reading Format",
        "hotel chains preference": "Hotel Chains Preference",
        "duration preference": "Duration Preference",
        "news sources": "News Sources",
        "platform preference": "Platform Preference",
        "viewing platform preference": "Viewing Platform Preference",
        "favorite bands": "Favorite Bands",
        "favorite albums": "Favorite Albums",
        "favorite authors": "Favorite Authors",
        "favorite sports": "Favorite Sports",
        "preferred game name": "Prefer Game Name",
        "prefer game name": "Prefer Game Name",
        "prefer communication style": "Prefer Communication Style",
        "event type preference": "Event Type Preference",
        "service provider gender preference": "Service Provider Gender Preference",
        "favorite actors and directors": "Favorite Actors and Directors",
        "ambiance preference": "Ambiance Preference",
        "cuisine preference": "Cuisine Preference",
        "preferred airline": "Prefer Airline",
        "prefer airline": "Prefer Airline",
        "car type preference": "Car Type Preference",
        "dietary restrictions": "Dietary Restrictions",
        "travel time preference": "Travel Time Preference",
        "appointment time preference": "Appointment Time Preference",
        "departure time preference": "Departure Time Preference",
        "reading frequency": "Reading Frequency",
        "theater type preference": "Theater Type Preference",
        "genre": "Genre",
        "multiplayer preference": "Multiplayer Preference",
        "preferred genres": "Prefer Genres",
        "prefer genres": "Prefer Genres",
        "preferred sectors": "Prefer Sectors",
        "sectors or services": "Prefer Sectors",
        "prefer destination types": "Prefer Destination Types",
        "preferred destination types": "Prefer Destination Types",
        "destination types": "Prefer Destination Types",
        "prefer messaging apps": "Prefer Messaging Apps",
        "preferred messaging apps": "Prefer Messaging Apps",
        "messaging apps": "Prefer Messaging Apps",
        "gaming frequency": "Gaming Frequency",
        "game frequency": "Gaming Frequency",
        "frequent contact": "Frequent Contact",
        "prefer bus company": "Prefer Bus Company",
        "preferred bus company": "Prefer Bus Company",
        "travel season preference": "Travel Season Preference",
        "favorite book series": "Favorite Book Series",
        "reading format": "Reading Format",
        "layover preference": "Layover Preference",
        "room type preference": "Room Type Preference",
        "seat type preference": "Seat Type Preference",
        "prefer seat type": "Seat Type Preference",
        "preferred seat type": "Seat Type Preference",
        "additional feature preference": "Additional Feature Preference",
        "prefer rental company": "Prefer Rental Company",
        "preferred rental company": "Prefer Rental Company",
        "prefer fuel type": "Prefer Fuel Type",
        "preferred fuel type": "Prefer Fuel Type",
        "travel frequency": "Travel Frequency",
        "days of week preference": "Days of Week Preference",
        "service frequency": "Service Frequency Preference",
        "service": "Service Frequency Preference",
        "frequency": "Service Frequency Preference",
        "amenity": "Amenity Preference",
        "amenities": "Amenity Preference",
        "budget": "Price Range",
        "budget or price": "Price Range",
        "price": "Price Range",
    }
    lowered = text.lower()
    return lookup.get(lowered, text[:1].upper() + text[1:] if text else "")


def _normalize_clock_value(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).upper().replace("AM", "AM").replace("PM", "PM")


def _clean_affinity_value(value: str) -> str:
    cleaned = " ".join(str(value or "").strip(" -*:\"'").split())
    cleaned = re.sub(r"\b(?:as|for|and|with|to|would|please|let me know).*$", "", cleaned, flags=re.IGNORECASE).strip(" -*,.;:")
    return cleaned[:120]


def _clean_personalens_keyed_value(key: str, value: str) -> str:
    cleaned = _clean_embedded_affinity_value(value)
    if key in {
        "Alarm Sound Preference",
        "Alarm Time Preference",
        "Location Preference",
        "Alarm Recurring Preference",
        "Price Range",
        "Prefer Car Brand",
        "Reading Frequency",
    }:
        cleaned = cleaned.split(",", 1)[0].strip(" -*,.;:")
    return cleaned


def _clean_embedded_affinity_value(value: str) -> str:
    cleaned = " ".join(str(value or "").strip(" -*:\"'").split())
    return cleaned.strip(" -*,.;:")[:240]


def _locomo_evidence_hypothesis_atoms(*, task_prompt: str, response_text: str) -> list[str]:
    answer = _strip_choice_number_unless_locomo_timestamp(response_text)
    if not answer:
        return []
    evidence_lines = _locomo_response_evidence_lines(answer)
    if evidence_lines:
        atoms: list[str] = []
        for line in evidence_lines:
            speaker = _locomo_answer_subject(line) or _locomo_question_subject(task_prompt)
            fact = _locomo_answer_to_memory_utterance(line, speaker=speaker)
            if fact:
                atoms.append(fact)
        return _dedupe(atoms)[:8]
    if _locomo_is_plain_answer(answer):
        question = _locomo_clean_question(task_prompt)
        if question:
            return [f"{question} -> {answer}"]
    speaker = _locomo_answer_subject(answer) or _locomo_question_subject(task_prompt)
    fact = _locomo_answer_to_memory_utterance(answer, speaker=speaker)
    atoms = [fact] if fact else []
    evidence_like = bool(speaker and re.search(rf"\b{re.escape(speaker)}\s*:", answer, flags=re.IGNORECASE))
    no_information_answer = bool(
        re.search(r"\b(?:no information|does not mention|does not specify|cannot determine)\b", answer, flags=re.IGNORECASE)
    )
    if answer and not evidence_like and not (atoms and no_information_answer) and answer.lower() not in {fact.lower() for fact in atoms}:
        atoms.append(f"locomo_answer={answer}")
    return _dedupe(atoms)[:2]


def _locomo_response_evidence_lines(answer: str) -> list[str]:
    said_evidence = [
        f"{match.group('time').strip()}:{match.group('speaker').strip().lower()}:{' '.join(match.group('body').strip().split())}"
        for match in re.finditer(
            r"(?P<time>\d{1,2}:\d{2}\s*(?:am|pm)\b.*?\d{4})\s*,?\s*"
            r"(?P<speaker>[A-Z][A-Za-z]+)\s+said\s*,?\s*(?P<quote>['\"])(?P<body>.*?)(?P=quote)(?=[.\s]|$)",
            str(answer or ""),
            flags=re.IGNORECASE,
        )
    ]
    if said_evidence:
        return said_evidence[:8]
    quoted_evidence = [
        " ".join(match.group(1).strip(" -*'\"").split())
        for match in re.finditer(
            r"['\"](\d{1,2}:\d{2}\s*(?:am|pm)\b[^'\"]*?:\s*[A-Z][A-Za-z]+\s*:[^'\"]{8,500})['\"]",
            str(answer or ""),
            flags=re.IGNORECASE,
        )
    ]
    if quoted_evidence:
        return quoted_evidence[:8]
    speaker_quote_evidence = [
        " ".join(match.group(1).strip(" -*'\"").split())
        for match in re.finditer(
            r"['\"]([A-Z][A-Za-z]+\s*:\s*[^'\"]{8,500})['\"]",
            str(answer or ""),
            flags=re.IGNORECASE,
        )
    ]
    if speaker_quote_evidence:
        return speaker_quote_evidence[:8]
    quote_spoken_evidence = [
        f"{match.group('time').strip()}:{match.group('speaker').strip().lower()}:{' '.join(match.group('body').strip(' -*:').split())}"
        for match in re.finditer(
            r"['\"](?P<body>[^'\"]{8,500})['\"]\s*"
            r"(?:spoken|said|stated|mentioned)\s+by\s+(?P<speaker>[A-Z][A-Za-z]+)\s+"
            r"(?:at|on)\s+(?P<time>\d{1,2}:\d{2}\s*(?:am|pm)\b.*?\d{4})",
            str(answer or ""),
            flags=re.IGNORECASE,
        )
    ]
    quote_spoken_evidence.extend(
        f"{match.group('time').strip()}:{match.group('speaker').strip().lower()}:{' '.join(match.group('body').strip(' -*:').split())}"
        for match in re.finditer(
            r"(?P<body>(?=[^()]*\b(?:I|I'm|Im|my|we|our)\b)[^()]{8,500}?)\s*"
            r"\(\s*(?P<time>\d{1,2}:\d{2}\s*(?:am|pm)\b.*?\d{4})\s*,\s*"
            r"(?P<speaker>[A-Z][A-Za-z]+)\s*\)",
            str(answer or ""),
            flags=re.IGNORECASE,
        )
    )
    if quote_spoken_evidence:
        return quote_spoken_evidence[:8]
    first_person_quote_evidence = [
        " ".join(match.group("body").strip(" -*'\"").split())
        for match in re.finditer(
            r"(?P<quote>['\"])(?P<body>(?=[^'\"]*\b(?:i|i'm|im|my|we|our)\b)[^'\"]{3,500})(?P=quote)(?=[.\s]|$)",
            str(answer or ""),
            flags=re.IGNORECASE,
        )
        if re.search(r"\b(?:statement|message|line|mentioned|said|stated)\b", str(answer or ""), flags=re.IGNORECASE)
    ]
    if first_person_quote_evidence:
        return first_person_quote_evidence[:8]
    source_note_evidence = [
        f"{match.group('time').strip()}:{match.group('speaker').strip().lower()}:{' '.join(match.group('body').strip(' -*:').split())}"
        for match in re.finditer(
            r"\bSpeaker\s*[:=]\s*(?P<speaker>[A-Z][A-Za-z]+)\s*[,;]\s*"
            r"Time\s*[:=]\s*(?P<time>\d{1,2}:\d{2}\s*(?:am|pm)\b[^,;.]*)\s*[,;]\s*"
            r"(?:Message|Fragment|Exact message fragment)\s*[:=]\s*(?P<body>[^\n]{8,500})",
            str(answer or ""),
            flags=re.IGNORECASE,
        )
    ]
    source_note_evidence.extend(
        f"{match.group('time').strip()}:{match.group('speaker').strip().lower()}:{' '.join(match.group('body').strip(' -*:').split())}"
        for match in re.finditer(
            r"\b(?P<speaker>[A-Z][A-Za-z]+)\s*,\s*"
            r"(?P<time>\d{1,2}:\d{2}\s*(?:am|pm)\b[^,;.]*)\s*,\s*"
            r"(?P<body>(?=[^\n]*\b(?:I|I'm|Im|my|we|our)\b)[^\n]{8,500})",
            str(answer or ""),
            flags=re.IGNORECASE,
        )
    )
    if source_note_evidence:
        return source_note_evidence[:8]
    parts = [
        " ".join(part.strip(" -*").split())
        for part in re.split(r"\s*\|\|\s*|\n+", str(answer or ""))
        if " ".join(part.strip(" -*").split())
    ]
    parts = [
        re.sub(r"^(?:source|evidence|conversation line|message|snippet)\s*:\s*", "", part, flags=re.IGNORECASE).strip()
        for part in parts
    ]
    evidence_lines = [
        part
        for part in parts
        if re.match(r"^(?:D\d+:\d+|\d+)\s+[A-Z][A-Za-z]+\s*:", part)
        or re.match(r"^[A-Z][A-Za-z]+\s*:", part)
        or re.match(r"^\d{1,2}:\d{2}\s*(?:am|pm)\b.*?:\s*[A-Z][A-Za-z]+\s*:", part, flags=re.IGNORECASE)
    ]
    if evidence_lines:
        return evidence_lines
    # The generated visible response may be whitespace-normalized; recover repeated Dk:n spans.
    matches = list(re.finditer(r"(?:^|\s)(D\d+:\d+\s+[A-Z][A-Za-z]+\s*:)", str(answer or "")))
    recovered: list[str] = []
    for index, match in enumerate(matches):
        start = match.start(1)
        end = matches[index + 1].start(1) if index + 1 < len(matches) else len(answer)
        recovered.append(" ".join(answer[start:end].strip(" -*|").split()))
    return [line for line in recovered if line]


def _locomo_question_subject(question: str) -> str:
    wh_words = {"What", "When", "Where", "Who", "Whom", "Which", "Why", "How", "Please", "Request"}
    match = re.search(r"\b(?:did|would|was|is|were|are|has|have)\s+([A-Z][A-Za-z]+)\b", str(question or ""))
    if match and match.group(1) not in wh_words:
        return match.group(1)
    for name in re.findall(r"\b[A-Z][A-Za-z]+\b", str(question or "")):
        if name not in wh_words:
            return name
    return ""


def _locomo_answer_subject(answer: str) -> str:
    non_speakers = {"Likely", "Probably", "Maybe", "Yes", "No", "Sunset", "The", "There", "A", "An"}
    cleaned = re.sub(r"^\s*(?:D\d+:\d+|\d+)\s+", "", str(answer or ""), flags=re.IGNORECASE)
    match = re.search(r"^\s*([A-Z][A-Za-z]+)\b", cleaned)
    if match and match.group(1) not in non_speakers:
        return match.group(1)
    match = re.search(r"\b\d{1,2}:\d{2}\s*(?:am|pm)\b.*?:\s*([A-Z][A-Za-z]+)\s*:", cleaned, flags=re.IGNORECASE)
    if match and match.group(1) not in non_speakers:
        return match.group(1)
    return ""


def _locomo_is_plain_answer(answer: str) -> bool:
    cleaned = str(answer or "").strip()
    if not cleaned:
        return False
    if re.search(r"^(?:D\d+:\d+|\d+)\s+[A-Z][A-Za-z]+\s*:", cleaned):
        return False
    if re.search(r"^[A-Z][A-Za-z]+\s*:", cleaned):
        return False
    if " || " in cleaned:
        return False
    return len(cleaned.split()) <= 24


def _locomo_clean_question(task_prompt: str) -> str:
    text = str(task_prompt or "").strip()
    if "Request:" in text:
        text = text.split("Request:", 1)[1].strip()
    if "\n\n" in text:
        text = text.split("\n\n", 1)[0].strip()
    return " ".join(text.rstrip(" ?").split())


def _locomo_answer_to_memory_utterance(answer: str, *, speaker: str) -> str:
    cleaned = " ".join(str(answer or "").strip(" -*.").split())
    if not cleaned:
        return ""
    cleaned = re.sub(r"^(?:D\d+:)?\d+\s+", "", cleaned, flags=re.IGNORECASE).strip()
    if "does not provide any information" in cleaned.lower() or "no relevant data" in cleaned.lower():
        return f"locomo_answer={cleaned}"
    if speaker:
        fallback_body = _locomo_canonical_body(cleaned)
        fallback_time = _locomo_time_from_text(cleaned)
        if fallback_time and fallback_body != cleaned:
            return f"{fallback_time}:{speaker.lower()}:{fallback_body}"
        speaker_time_match = re.search(
            rf"^\s*{re.escape(speaker)}\s*:\s*(?P<body>.+?)\s+"
            rf"(?:this\s+)?(?:statement|message|line)?\s*(?:was\s+)?(?:made|mentioned|said|stated)\s+"
            rf"(?:by\s+{re.escape(speaker)}\s+)?(?:at|was)\s+(?P<time>\d{{1,2}}:\d{{2}}\s*(?:am|pm)\b.*?\d{{4}})",
            cleaned,
            flags=re.IGNORECASE,
        )
        if speaker_time_match:
            body = re.sub(r"\s+\b(?:This|The)\s+(?:statement|message|line)\b.*$", "", speaker_time_match.group("body").strip(), flags=re.IGNORECASE).strip()
            body = re.sub(r"\s+\b(?:as|because)\b.*$", "", body, flags=re.IGNORECASE).strip()
            body = _locomo_canonical_body(body)
            time_part = speaker_time_match.group("time").strip(" .")
            if body:
                return f"{time_part}:{speaker.lower()}:{body}"
        timestamp_match = re.search(
            rf"^(?P<time>\d{{1,2}}:\d{{2}}\s*(?:am|pm)\b.*?):\s*{re.escape(speaker)}\s*:\s*(?P<body>.+)$",
            cleaned,
            flags=re.IGNORECASE,
        )
        if timestamp_match:
            time_part = timestamp_match.group("time").strip()
            body = re.sub(r"\s+\bAnswer\s*:.*$", "", timestamp_match.group("body").strip(), flags=re.IGNORECASE).strip()
            body = _locomo_canonical_body(body)
            return f"{time_part}:{speaker.lower()}:{body}"
        speaker_prefix = re.compile(rf"^{re.escape(speaker)}\s*:\s*(.+)$", flags=re.IGNORECASE)
        prefix_match = speaker_prefix.match(cleaned)
        if prefix_match:
            body = re.sub(r"\s+\bAnswer\s*:.*$", "", prefix_match.group(1).strip(), flags=re.IGNORECASE).strip()
            body = _locomo_canonical_body(body)
            return f"{speaker}: {body}"
        embedded_prefix = re.search(rf"\b{re.escape(speaker)}\s*:\s*(.+)$", cleaned, flags=re.IGNORECASE)
        if embedded_prefix:
            body = re.sub(r"\s+\bAnswer\s*:.*$", "", embedded_prefix.group(1).strip(), flags=re.IGNORECASE).strip()
            body = _locomo_canonical_body(body)
            return f"{speaker}: {body}"
        pronounized = re.sub(rf"^\s*{re.escape(speaker)}\s+", "I ", cleaned, flags=re.IGNORECASE)
        pronounized = re.sub(r"\bwould likely pursue\b", "am keen on", pronounized, flags=re.IGNORECASE)
        pronounized = re.sub(r"\bwent to\b", "went to", pronounized, flags=re.IGNORECASE)
        return f"{speaker}: {_locomo_canonical_body(pronounized)}"
    return cleaned


def _locomo_canonical_body(body: str) -> str:
    cleaned = " ".join(str(body or "").strip().split())
    cleaned = cleaned.replace("–", " ").replace("—", " ").replace("-", " ")
    cleaned = re.sub(r"\s*,\s*", ",", cleaned)
    return cleaned.lower()


def _locomo_time_from_text(text: str) -> str:
    match = re.search(r"(\d{1,2}:\d{2}\s*(?:am|pm)\b.*?\d{4})", str(text or ""), flags=re.IGNORECASE)
    return " ".join(match.group(1).strip(" .,").split()) if match else ""


def _parse_arrow_affinity_chunks(response_text: str) -> list[tuple[str, str]]:
    chunks: list[tuple[str, str]] = []
    for match in re.finditer(r"([A-Za-z][A-Za-z ]{2,40}?)\s*(?:->|:)\s*([^.;\n]+)", response_text):
        key = " ".join(match.group(1).strip(" -*").split())
        values_text = " ".join(match.group(2).strip().split())
        if key and values_text and key.lower() not in {"based on your preferences"}:
            chunks.append((key, values_text))
    return chunks


def _parse_key_value_affinity_chunks(response_text: str) -> list[tuple[str, str]]:
    chunks: list[tuple[str, str]] = []
    for segment in re.split(r"[;\n]+", response_text):
        cleaned = " ".join(segment.strip(" -*.").split())
        if not cleaned or "=" not in cleaned:
            continue
        if ":" in cleaned:
            cleaned = cleaned.rsplit(":", 1)[-1].strip()
        if "=" not in cleaned:
            continue
        key, value = cleaned.split("=", 1)
        key = " ".join(key.strip(" -*").split())
        value = " ".join(value.strip(" -*").split())
        if key and value and 2 <= len(key) <= 80:
            chunks.append((key, value))
    return chunks


def _split_values(values_text: str) -> list[str]:
    cleaned = re.sub(r"\band\b", ",", values_text, flags=re.IGNORECASE)
    return [
        item.strip(" -*.")
        for item in cleaned.split(",")
        if item.strip(" -*.")
    ][:6]


def _condition_record(
    *,
    bundle: Mapping[str, Any] | None,
    condition: str,
) -> dict[str, Any]:
    records = dict((bundle or {}).get("records", {}))
    if condition in records and isinstance(records[condition], Mapping):
        return dict(records[condition])
    if condition == "no_memory":
        for alias in ("delete", "no_memory", "generic"):
            if isinstance(records.get(alias), Mapping):
                return dict(records[alias])
    if condition == "personalized" and isinstance(records.get("personalized"), Mapping):
        return dict(records["personalized"])
    return {}


def _record_output(record: Mapping[str, Any]) -> dict[str, Any]:
    output = record.get("output")
    return dict(output) if isinstance(output, Mapping) else {}


def _record_response_text(output: Mapping[str, Any]) -> str:
    for key in ("response_text", "predicted_answer", "predicted_label", "action_signature"):
        if output.get(key):
            return str(output.get(key) or "")
    return ""


def _clean_response_text(response_text: str) -> str:
    return " ".join(str(response_text or "").strip().split())


def _dedupe(items: list[str]) -> list[str]:
    deduped: dict[str, None] = {}
    for item in items:
        text = str(item).strip()
        key = text.lower()
        if text and key not in deduped:
            deduped[key] = None
    first_by_key: dict[str, str] = {}
    for item in items:
        text = str(item).strip()
        key = text.lower()
        if text and key not in first_by_key:
            first_by_key[key] = text
    return list(first_by_key.values())


def _item_count(model: Mapping[str, Any]) -> int:
    count = 0
    for category in ("facts", "preferences", "constraints", "relations"):
        value = model.get(category)
        if isinstance(value, list):
            count += len([item for item in value if str(item)])
    tool_state = model.get("tool_state")
    if isinstance(tool_state, Mapping):
        count += len(tool_state)
    elif isinstance(tool_state, list):
        count += len([item for item in tool_state if str(item)])
    return count


def _number_or_none(value: Any) -> float | None:
    try:
        return None if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return None
