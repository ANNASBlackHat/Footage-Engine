"""Unit tests for CueResolver in footage-engine."""

import pytest
from footage_engine.retrieval.cue_resolver import (
    parse_time_range,
    parse_asset_list_line,
    resolve_cue,
    CueDefinition,
)


def test_parse_time_range():
    start, end = parse_time_range("0:28–0:36")
    assert start == 28.0
    assert end == 36.0

    start_m, end_m = parse_time_range("18:36–19:35")
    assert start_m == 18 * 60 + 36.0
    assert end_m == 19 * 60 + 35.0


def test_parse_asset_list_line_rating():
    line = "| 0:28–0:36 | **Rating key card** — five tiers, slam in | Build | 🔵 CREATED |"
    cue = parse_asset_list_line(line)
    assert cue is not None
    assert cue.start_sec == 28.0
    assert cue.end_sec == 36.0
    assert "Rating key card" in cue.asset_description
    assert cue.rights == "CREATED"


def test_resolve_cue_motion_components():
    # 1. Rating Card
    cue1 = parse_asset_list_line("| 0:28–0:36 | **Rating key card** — five tiers, slam in | Build | 🔵 CREATED |")
    resolved1 = resolve_cue(cue1)
    assert resolved1.cue_type == "motion_graphic"
    assert resolved1.component_id == "Evidence/RatingCard"

    # 2. Sourcing Card
    cue2 = parse_asset_list_line("| 1:00–1:20 | Four-tier sourcing card | Build | 🔵 CREATED |")
    resolved2 = resolve_cue(cue2)
    assert resolved2.cue_type == "motion_graphic"
    assert resolved2.component_id == "Evidence/SourcingCard"

    # 3. Measurement Compare
    cue3 = parse_asset_list_line("| 6:20–6:40 | **19 ft → 17 ft → 13 ft 1 in** — the shrinkage graphic | Build | 🔵 CREATED |")
    resolved3 = resolve_cue(cue3)
    assert resolved3.cue_type == "motion_graphic"
    assert resolved3.component_id == "DataAnimations/MeasurementCompare"

    # 4. Reprint Chain
    cue4 = parse_asset_list_line("| 11:20–11:40 | **The reprint chain graphic**, building left to right | Build | 🔵 CREATED |")
    resolved4 = resolve_cue(cue4)
    assert resolved4.cue_type == "motion_graphic"
    assert resolved4.component_id == "Evidence/ReprintChain"

    # 5. Verdict Table
    cue5 = parse_asset_list_line("| 18:36–18:50 | **Full rating table**, one row per line as read | Build | 🔵 CREATED |")
    resolved5 = resolve_cue(cue5)
    assert resolved5.cue_type == "motion_graphic"
    assert resolved5.component_id == "Evidence/VerdictTable"

    # 6. Archival Document
    cue6 = parse_asset_list_line("| 1:20–1:35 | Slow drift across the *Pearl* account text, one phrase highlighted | *Homeward Mail* scan | 🟡 CHECK |")
    resolved6 = resolve_cue(cue6)
    assert resolved6.cue_type == "archival_document"
    assert resolved6.component_id == "Archival/DocumentViewer"

    # 7. Stock B-roll
    cue7 = parse_asset_list_line("| 4:50–5:10 | Black water, rope with nothing on it | Own footage | 🔵 CREATED |")
    resolved7 = resolve_cue(cue7)
    assert resolved7.cue_type == "footage_search"
    assert "water" in resolved7.search_query.lower()
