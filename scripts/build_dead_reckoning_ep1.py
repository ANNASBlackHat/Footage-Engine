#!/usr/bin/env python3
"""Build and compile the complete Dead Reckoning Episode 1 Timeline from asset_list.md using Qwen worker."""

import json
import os
import sys
from pathlib import Path

from footage_engine.retrieval.cue_resolver import (
    parse_asset_list_line,
    resolve_cue,
    CueDefinition,
)
from footage_engine.models.db import get_session_factory
from footage_engine.models.media import MediaItem
from footage_engine.worker import submit_search_footage, wait_for_job

ASSET_LIST_PATH = Path("/Users/annasblackhat/Documents/Experiment/content-expert/projects/1.-The-Giant-Squid-That-San-a-Ship/asset_list.md")
OUTPUT_PROJECT_PATH = Path("/Users/annasblackhat/Documents/Experiment/content-expert/projects/1.-The-Giant-Squid-That-San-a-Ship/timeline.json")
FRONTEND_PUBLIC_PATH = Path("/Users/annasblackhat/Documents/Experiment/video-generation-frontend/public/dead_reckoning_ep1_timeline.json")

# Default fallback if a query returns 0 results
FALLBACK_VIDEO_URL = "https://videos.pexels.com/video-files/35698367/15129548_640_360_25fps.mp4"


def compile_episode_timeline(use_worker: bool = True):
    print(f"Reading asset list from: {ASSET_LIST_PATH}")
    if not ASSET_LIST_PATH.exists():
        print(f"Error: Asset list file not found at {ASSET_LIST_PATH}")
        sys.exit(1)

    with open(ASSET_LIST_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    cues: list[CueDefinition] = []
    for line in lines:
        cue = parse_asset_list_line(line)
        if cue:
            cues.append(resolve_cue(cue))

    print(f"Parsed {len(cues)} visual cues across the episode.")

    video_items = []
    text_motion_items = []
    total_duration = 0.0

    # Cache for B-roll searches so duplicate/similar queries don't re-query
    search_cache: dict[str, list[dict]] = {}

    for i, cue in enumerate(cues):
        clip_id = f"cue_{i+1:02d}"
        duration = max(0.5, cue.end_sec - cue.start_sec)
        if cue.end_sec > total_duration:
            total_duration = cue.end_sec

        if cue.cue_type == "motion_graphic":
            text_motion_items.append({
                "id": f"motion_{clip_id}",
                "trackStart": cue.start_sec,
                "trackEnd": cue.end_sec,
                "duration": duration,
                "componentId": cue.component_id,
                "props": cue.props,
                "content": cue.asset_description,
                "layoutRole": "takeover",
            })
        elif cue.cue_type == "archival_document":
            # DocumentViewer is a motion component that displays the document
            doc_props = dict(cue.props)
            doc_props["durationInFrames"] = int(duration * 30)
            text_motion_items.append({
                "id": f"doc_{clip_id}",
                "trackStart": cue.start_sec,
                "trackEnd": cue.end_sec,
                "duration": duration,
                "componentId": "Archival/DocumentViewer",
                "props": doc_props,
                "content": cue.asset_description,
                "layoutRole": "takeover",
            })
        else:
            # Footage search / B-roll via Qwen worker
            query = cue.search_query or "ocean waves calm sea"
            print(f"\n[{clip_id}] Resolving B-roll query: '{query}' ({duration:.1f}s)")

            results = []
            if use_worker:
                if query in search_cache:
                    results = search_cache[query]
                    print(f"  ↳ Using cached search results ({len(results)} candidates)")
                else:
                    try:
                        print("  ↳ Enqueueing search job to Qwen GPU worker...")
                        job_id = submit_search_footage(query, top_k=5, backend="qwen")
                        job = wait_for_job(job_id=job_id, timeout_sec=45)
                        if job and job.get("status") == "done" and job.get("result"):
                            results = job["result"].get("results", [])
                            search_cache[query] = results
                            print(f"  ↳ Worker returned {len(results)} matches!")
                    except Exception as e:
                        print(f"  ⚠️ Worker search failed ({e}), using fallback")

            # Select best match or fallback
            selected_url = FALLBACK_VIDEO_URL
            source_in = 0.0
            source_out = duration
            score = 0.0

            if results:
                best = results[0]
                selected_url = best.get("storage_url") or best.get("source_url") or FALLBACK_VIDEO_URL
                source_in = float(best.get("start_ts", 0.0))
                source_out = source_in + duration
                score = float(best.get("score", 0.0))
                print(f"  ✅ Selected Match: {selected_url} (Score: {score:.4f}, [{source_in}s - {source_out}s])")
            else:
                print(f"  ⚠️ No worker results returned, assigned fallback URL: {selected_url}")

            video_items.append({
                "id": f"broll_{clip_id}",
                "trackStart": cue.start_sec,
                "trackEnd": cue.end_sec,
                "duration": duration,
                "assetType": "video",
                "storagePath": selected_url,
                "storageUrl": selected_url,
                "sourceIn": source_in,
                "sourceOut": source_out,
                "score": score,
                "searchQuery": query,
                "description": cue.asset_description,
            })

    timeline_data = {
        "jobId": "dead_reckoning_ep1",
        "fps": 30,
        "total_duration": total_duration,
        "width": 1920,
        "height": 1080,
        "orientation": "horizontal",
        "tracks": [
            {
                "id": "video",
                "label": "B-Roll Footage Track",
                "type": "video",
                "items": video_items,
            },
            {
                "id": "text",
                "label": "Documentary Motion & Archival Track",
                "type": "text",
                "items": text_motion_items,
            },
        ],
        "metadata": {
            "title": "Dead Reckoning — The Giant Squid That Sank a Ship",
            "cue_count": len(cues),
            "motion_count": len(text_motion_items),
            "broll_count": len(video_items),
            "runtime_formatted": f"{int(total_duration // 60)}:{int(total_duration % 60):02d}",
        },
    }

    # Save to content project directory
    with open(OUTPUT_PROJECT_PATH, "w", encoding="utf-8") as f:
        json.dump(timeline_data, f, indent=2)
    print(f"\nSaved compiled episode timeline to: {OUTPUT_PROJECT_PATH}")

    # Ensure frontend public directory exists and write copy
    FRONTEND_PUBLIC_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FRONTEND_PUBLIC_PATH, "w", encoding="utf-8") as f:
        json.dump(timeline_data, f, indent=2)
    print(f"Published to frontend preview at: {FRONTEND_PUBLIC_PATH}")

    print(f"\nEpisode Summary:")
    print(f"- Total Runtime: {timeline_data['metadata']['runtime_formatted']} ({total_duration:.1f}s)")
    print(f"- Total Cues: {len(cues)}")
    print(f"- Documentary Motion & Archival Items: {len(text_motion_items)}")
    print(f"- B-Roll Video Tracks: {len(video_items)}")
    print("✅ Build complete!")


if __name__ == "__main__":
    compile_episode_timeline(use_worker=True)
