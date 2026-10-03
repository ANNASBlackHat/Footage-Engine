"""Cue Resolver: maps documentary script cues and asset_list entries to motion components or footage search queries."""

import re
from dataclasses import dataclass, field
from typing import Optional, Any


@dataclass
class CueDefinition:
    time_range: str
    start_sec: float
    end_sec: float
    asset_description: str
    source: str
    rights: str
    cue_type: str = "unknown"  # "motion_graphic" | "archival_document" | "footage_search"
    component_id: Optional[str] = None
    search_query: Optional[str] = None
    props: dict[str, Any] = field(default_factory=dict)


def parse_time_range(range_str: str) -> tuple[float, float]:
    """Parses a time string like '0:28–0:36' or '18:36–19:35' into (start_sec, end_sec)."""
    clean = range_str.replace("–", "-").replace("—", "-").strip()
    parts = clean.split("-")
    if len(parts) != 2:
        return 0.0, 10.0

    def to_sec(s: str) -> float:
        s = s.strip()
        sub = s.split(":")
        if len(sub) == 2:
            return float(sub[0]) * 60 + float(sub[1])
        elif len(sub) == 3:
            return float(sub[0]) * 3600 + float(sub[1]) * 60 + float(sub[2])
        return float(s)

    return to_sec(parts[0]), to_sec(parts[1])


def parse_asset_list_line(line: str) -> Optional[CueDefinition]:
    """Parses a single markdown table line from asset_list.md."""
    clean = line.strip()
    if not clean.startswith("|") or "---" in clean or "Asset" in clean or "Time" in clean:
        return None

    columns = [c.strip() for c in clean.split("|")]
    # Columns usually: ['', '0:00–0:08', 'Asset desc', 'Source', 'Rights', '']
    valid_cols = [c for c in columns if c]
    if len(valid_cols) < 4:
        return None

    time_str = valid_cols[0]
    if not re.match(r"^\d+:\d+", time_str):
        return None

    asset_desc = valid_cols[1].strip("* ")
    source = valid_cols[2]
    rights_raw = valid_cols[3]

    # Extract rights tag
    rights = "CHECK"
    if "PD" in rights_raw:
        rights = "PD"
    elif "CREATED" in rights_raw:
        rights = "CREATED"
    elif "LICENSED" in rights_raw:
        rights = "LICENSED"

    start_sec, end_sec = parse_time_range(time_str)

    return CueDefinition(
        time_range=time_str,
        start_sec=start_sec,
        end_sec=end_sec,
        asset_description=asset_desc,
        source=source,
        rights=rights,
    )


def resolve_cue(cue: CueDefinition) -> CueDefinition:
    """Classifies a cue into a motion component or a footage search query."""
    desc = cue.asset_description.lower()
    src = cue.source.lower()

    # 1. Rating Card
    if "rating key card" in desc or "rating table" in desc and "full" not in desc:
        cue.cue_type = "motion_graphic"
        cue.component_id = "Evidence/RatingCard"
        cue.props = {
            "title": "DEAD RECKONING EVIDENCE RATING",
            "rating": "UNSUPPORTED",
            "claim": cue.asset_description,
        }
        return cue

    # 2. Sourcing Card
    if "sourcing card" in desc or "four-tier" in desc:
        cue.cue_type = "motion_graphic"
        cue.component_id = "Evidence/SourcingCard"
        cue.props = {
            "tier": 1,
            "source": cue.source,
        }
        return cue

    # 3. Measurement Comparisons / Shrinkage
    if "shrinkage" in desc or "→" in cue.asset_description or "->" in cue.asset_description or "three-way size" in desc or "scale graphic" in desc or "55 ft" in desc:
        cue.cue_type = "motion_graphic"
        cue.component_id = "DataAnimations/MeasurementCompare"
        cue.props = {
            "title": "SPECIMEN MEASUREMENT COLLAPSE",
            "steps": [
                {"label": "Documented", "value": "19 ft"},
                {"label": "Preserved", "value": "17 ft"},
                {"label": "Desiccated", "value": "13 ft 1 in"},
            ],
        }
        return cue

    # 4. Reprint Chain
    if "reprint chain" in desc:
        cue.cue_type = "motion_graphic"
        cue.component_id = "Evidence/ReprintChain"
        cue.props = {
            "title": "NEWSPAPER REPRINT TRANSMISSION",
            "nodes": [
                {"outlet": "Homeward Mail", "date": "29 June 1874"},
                {"outlet": "The Times", "date": "4 July 1874"},
                {"outlet": "News of the World", "date": "5 July 1874"},
                {"outlet": "Sacramento Daily Union", "date": "31 July 1874"},
            ],
        }
        return cue

    # 5. Full Verdict Table
    if "verdict" in desc or "full rating table" in desc:
        cue.cue_type = "motion_graphic"
        cue.component_id = "Evidence/VerdictTable"
        cue.props = {
            "title": "DEAD RECKONING — INVESTIGATIVE VERDICT",
        }
        return cue

    # 6. Archival Documents & Scans (Newspapers, Lloyd's register, engravings)
    if "scan" in src or "register" in desc or "newspaper" in desc or "engraving" in desc or "plate" in desc or "photograph" in desc or "lloyd's" in desc or "homeward mail" in desc or "times" in desc:
        cue.cue_type = "archival_document"
        cue.component_id = "Archival/DocumentViewer"
        cue.props = {
            "title": cue.source.upper() if len(cue.source) < 40 else "ARCHIVAL DOCUMENT",
            "highlightText": cue.asset_description[:50],
        }
        return cue

    # 7. Footage / B-roll Search
    cue.cue_type = "footage_search"
    clean_query = re.sub(r"[\*\_\[\]]", "", cue.asset_description)
    cue.search_query = clean_query
    return cue
