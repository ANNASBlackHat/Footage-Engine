"""OpenAI-compatible LLM client for Multi-Query Expansion and Optional Reranking."""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import requests

from footage_engine.config import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass
class ExpandedBeat:
    """Decomposed script beat containing visual search variants and optional detected entity."""

    raw_beat: str
    detected_entity: Optional[str] = None
    queries: list[str] = field(default_factory=list)


class LLMClient:
    """Lightweight HTTP client for any OpenAI-compatible Chat Completions endpoint."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()

    def is_available(self) -> bool:
        """Returns True if an API key is configured."""
        return bool(self.settings.effective_llm_api_key)

    def complete_json(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant that outputs only valid JSON.",
        timeout: float = 30.0,
    ) -> Optional[dict[str, Any]]:
        """Sends a prompt to the OpenAI-compatible endpoint and parses the JSON response."""
        api_key = self.settings.effective_llm_api_key
        if not api_key:
            logger.debug("No LLM API key configured. Skipping LLM request.")
            return None

        base_url = self.settings.effective_llm_base_url
        url = f"{base_url}/chat/completions"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.settings.LLM_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            raw_text = data["choices"][0]["message"]["content"].strip()

            # Clean potential markdown code blocks (e.g. ```json ... ```)
            clean_text = raw_text
            if "```" in clean_text:
                match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean_text)
                if match:
                    clean_text = match.group(1).strip()

            return json.loads(clean_text)
        except Exception as e:
            logger.warning(f"LLM completion request failed: {e}")
            return None


class QueryExpander:
    """Expands narrative voiceover script beats into concrete visual search queries."""

    def __init__(self, client: Optional[LLMClient] = None):
        self.client = client or LLMClient()

    def expand_beat(self, beat_text: str) -> ExpandedBeat:
        """Extracts canonical entity and decomposes beat into 2-4 visual search queries."""
        clean_text = beat_text.strip()
        if not clean_text:
            return ExpandedBeat(raw_beat="")

        if not self.client.is_available():
            # Graceful fallback without LLM: use raw beat as single query
            return ExpandedBeat(raw_beat=clean_text, queries=[clean_text])

        system_prompt = (
            "You are a professional video editor's search assistant. "
            "You translate voiceover script sentences into concrete visual search queries for stock video search. "
            "Respond ONLY with a valid JSON object."
        )

        user_prompt = f"""Given this video voiceover beat:
"{clean_text}"

Extract the following in JSON format:
1. "detected_entity": The specific historical ship, person, animal, vehicle, or landmark mentioned in the beat, or null if general.
2. "queries": A list of 2 to 4 concise visual search phrases (3-6 words each) describing what the camera should see:
   - Physical objects and actions only (e.g. "naval ship in heavy storm", "turbulent dark ocean waves").
   - Strip dates, abstract storytelling metaphors, and voiceover filler.

Example JSON output:
{{
  "detected_entity": "USS Cyclops",
  "queries": [
    "vintage naval cargo ship sailing",
    "rough dark stormy ocean waves",
    "military vessel cruising in open sea"
  ]
}}"""

        data = self.client.complete_json(prompt=user_prompt, system_prompt=system_prompt)
        if not data or not isinstance(data, dict):
            return ExpandedBeat(raw_beat=clean_text, queries=[clean_text])

        detected = data.get("detected_entity")
        if detected and isinstance(detected, str):
            detected = detected.strip()
            if detected.lower() in ("null", "none", ""):
                detected = None

        queries = data.get("queries") or []
        if isinstance(queries, list):
            clean_queries = [str(q).strip() for q in queries if str(q).strip()]
        else:
            clean_queries = []

        if not clean_queries:
            clean_queries = [clean_text]

        return ExpandedBeat(
            raw_beat=clean_text,
            detected_entity=detected,
            queries=clean_queries,
        )


class LLMJudge:
    """Lightweight LLM judge to rank candidate clips against the original narrative beat."""

    def __init__(self, client: Optional[LLMClient] = None):
        self.client = client or LLMClient()

    def judge_and_rerank(
        self,
        beat_text: str,
        candidates: list[Any],  # list[ChunkResult]
        top_n: int = 5,
    ) -> list[Any]:
        """Prompts the LLM to select and rank the best matching candidate clips."""
        if not candidates or not self.client.is_available():
            return candidates

        pool = candidates[:top_n]
        summaries = []
        for i, c in enumerate(pool, 1):
            desc_parts = []
            if getattr(c, "entity_name", None):
                desc_parts.append(f"Entity: {c.entity_name}")
            if getattr(c, "caption", None):
                desc_parts.append(f"Caption: {c.caption}")
            title = c.item_metadata.get("title") if hasattr(c, "item_metadata") and c.item_metadata else None
            if title:
                desc_parts.append(f"Title: {title}")
            if getattr(c, "tags", None):
                desc_parts.append(f"Tags: {', '.join(c.tags[:5])}")
            desc_str = " | ".join(desc_parts) or f"{c.provider} {c.media_type}"
            summaries.append(f"[{i}] Chunk ID: {c.chunk_id} | {desc_str}")

        system_prompt = (
            "You are an expert documentary film editor. "
            "You pick the best visual footage clip to accompany a narration voiceover beat. "
            "Respond ONLY with a valid JSON object."
        )

        user_prompt = f"""Voiceover beat:
"{beat_text}"

Candidate video clips:
{chr(10).join(summaries)}

Rank these candidate clips from best visual fit to worst visual fit.
Return JSON with:
1. "ranked_chunk_ids": List of chunk IDs in order of preference (best first).
2. "confidence_scores": Map of chunk_id to confidence score between 0.0 and 1.0.

Example:
{{
  "ranked_chunk_ids": ["id_2", "id_1"],
  "confidence_scores": {{"id_2": 0.92, "id_1": 0.65}}
}}"""

        data = self.client.complete_json(prompt=user_prompt, system_prompt=system_prompt)
        if not data or not isinstance(data, dict):
            return candidates

        ranked_ids = data.get("ranked_chunk_ids") or []
        scores_map = data.get("confidence_scores") or {}

        if not ranked_ids:
            return candidates

        # Re-sort pool based on ranked_ids
        cand_map = {c.chunk_id: c for c in candidates}
        ordered: list[Any] = []
        seen = set()

        for cid in ranked_ids:
            if cid in cand_map and cid not in seen:
                c = cand_map[cid]
                # Optionally override score if provided by LLM
                if cid in scores_map and isinstance(scores_map[cid], (int, float)):
                    c.score = float(scores_map[cid])
                ordered.append(c)
                seen.add(cid)

        # Append any unranked remaining candidates
        for c in candidates:
            if c.chunk_id not in seen:
                ordered.append(c)

        return ordered
