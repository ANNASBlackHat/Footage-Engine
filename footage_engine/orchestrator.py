"""Ingestion orchestrator with pre-spend duplicate detection."""

import logging
import os
import tempfile
import urllib.parse
import uuid
from typing import Any, Literal, Optional
import requests
from sqlalchemy import or_, select

from footage_engine.config import Settings, get_settings
from footage_engine.models.db import get_db_session, init_db
from footage_engine.models.media import MediaItem, MediaStatus, MediaType
from footage_engine.sources.base import Candidate, SourceAdapter
from footage_engine.sources.coverr import CoverrAdapter
from footage_engine.sources.direct import DirectURLAdapter
from footage_engine.sources.pexels import PexelsAdapter
from footage_engine.sources.pixabay import PixabayAdapter
from footage_engine.sources.protocol import URLProvider
from footage_engine.sources.youtube import (
    YouTubeAdapter,
    extract_youtube_video_id,
    is_youtube_url,
    normalize_youtube_url,
)
from footage_engine.storage import get_storage_backend
from footage_engine.storage.base import StorageBackend

logger = logging.getLogger(__name__)


class Orchestrator:
    """Coordinates ingestion, pre-spend dedup, raw downloading, and storage."""

    def __init__(
        self,
        settings: Settings | None = None,
        storage: StorageBackend | None = None,
        database_url: str | None = None,
    ):
        self.settings = settings or get_settings()
        self.storage = storage or get_storage_backend(self.settings)
        self.database_url = database_url or self.settings.DATABASE_URL
        init_db(self.database_url)

        # Register provider adapters
        self.adapters: dict[str, SourceAdapter] = {
            "pixabay": PixabayAdapter(api_key=self.settings.PIXABAY_API_KEY),
            "pexels": PexelsAdapter(api_key=self.settings.PEXELS_API_KEY),
            "coverr": CoverrAdapter(api_key=self.settings.COVERR_API_KEY),
            "youtube": YouTubeAdapter(),
        }

    def _normalize_url(self, url: str) -> str:
        """Strip trailing slashes or normalize tracking parameters if needed."""
        if is_youtube_url(url):
            return normalize_youtube_url(url)
        parsed = urllib.parse.urlparse(url.strip())
        return urllib.parse.urlunparse(parsed)

    def find_existing(
        self,
        source_url: str,
        provider: str,
        source_id: Optional[str] = None,
    ) -> Optional[MediaItem]:
        """Check database for exact duplicate match BEFORE any network download or storage write."""
        normalized_url = self._normalize_url(source_url)
        with get_db_session(self.database_url) as session:
            stmt = select(MediaItem).where(
                or_(
                    (MediaItem.provider == provider) & (MediaItem.source_id == source_id)
                    if source_id is not None
                    else False,
                    MediaItem.source_url == normalized_url,
                )
            )
            item = session.execute(stmt).scalars().first()
            if item:
                # Eagerly load attributes before session closes
                session.expunge(item)
                return item
        return None

    def ingest(
        self,
        source_url: str,
        provider: str = "manual",
        source_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        license_type: str = "unknown",
        media_type: str = "video",
        duration_sec: Optional[float] = None,
        resolution: Optional[str] = None,
    ) -> MediaItem:
        """Idempotent ingestion. Returns existing MediaItem immediately if already ingested."""
        normalized_url = self._normalize_url(source_url)
        if is_youtube_url(normalized_url):
            if provider in ("manual", "direct", "unknown"):
                provider = "youtube"
            if not source_id:
                source_id = extract_youtube_video_id(normalized_url)
            media_type = "video"
            if not metadata or not duration_sec or not resolution:
                try:
                    yt_adapter = self.adapters.get("youtube") or YouTubeAdapter()
                    cand = yt_adapter.url_to_candidate(normalized_url, metadata=metadata)
                    metadata = cand.metadata
                    duration_sec = duration_sec or cand.duration_sec
                    resolution = resolution or cand.resolution
                    if license_type == "unknown":
                        license_type = cand.license_type
                except Exception as e:
                    logger.warning(f"Could not auto-extract YouTube metadata for {normalized_url}: {e}")

        # 1. Dedup check (Pre-spend)
        existing = self.find_existing(normalized_url, provider, source_id)
        if existing:
            logger.info(f"Duplicate found for {provider}:{source_id or normalized_url}. Returning existing MediaItem {existing.id}.")
            return existing

        storage_path = normalized_url

        # 2. Download and upload raw file ONLY if UPLOAD_RAW_TO_STORAGE is enabled
        if self.settings.UPLOAD_RAW_TO_STORAGE:
            print(f"  → Downloading from {provider} ({source_id or 'direct'})...", flush=True)
            try:
                if provider == "youtube" or is_youtube_url(normalized_url):
                    yt_adapter = self.adapters.get("youtube") or YouTubeAdapter()
                    content = yt_adapter.download_bytes(normalized_url)
                elif normalized_url.startswith("file://"):
                    local_fpath = urllib.parse.unquote(normalized_url[7:])
                    with open(local_fpath, "rb") as f:
                        content = f.read()
                else:
                    headers = {
                        "User-Agent": "FootageEngine/0.1 (+https://github.com/footage-engine)"
                    }
                    resp = requests.get(normalized_url, stream=True, timeout=60, headers=headers)
                    resp.raise_for_status()
                    content = resp.content
                print(f"    Downloaded {len(content) / (1024*1024):.2f} MB.", flush=True)
            except Exception as e:
                logger.error(f"Failed to download raw asset from {normalized_url}: {e}")
                with get_db_session(self.database_url) as session:
                    failed_item = MediaItem(
                        provider=provider,
                        source_id=source_id,
                        source_url=normalized_url,
                        license_type=license_type,
                        media_type=MediaType.IMAGE if media_type == "image" else MediaType.VIDEO,
                        storage_path="",
                        status=MediaStatus.FAILED,
                        error_message=f"Download failed: {str(e)}",
                        item_metadata=metadata or {},
                    )
                    session.add(failed_item)
                    session.flush()
                    session.refresh(failed_item)
                    session.expunge(failed_item)
                    return failed_item

            # Determine filename and extension
            clean_path = urllib.parse.urlparse(normalized_url).path
            ext = os.path.splitext(clean_path)[1]
            if not ext:
                ext = ".mp4" if media_type == "video" else ".jpg"

            file_uuid = str(uuid.uuid4())
            safe_source_id = source_id.replace("/", "_") if source_id else file_uuid[:8]
            filename = f"{provider}_{safe_source_id}_{file_uuid[:8]}{ext}"

            # 3. Upload to storage backend
            print(f"  → Uploading to {self.settings.STORAGE_BACKEND} storage ({filename})...", flush=True)
            storage_path = self.storage.save_file(content, filename)
            print(f"    Stored at: {storage_path}", flush=True)

        # 4. Insert media_item row (status=pending)
        with get_db_session(self.database_url) as session:
            media_item = MediaItem(
                provider=provider,
                source_id=source_id,
                source_url=normalized_url,
                license_type=license_type,
                media_type=MediaType.IMAGE if media_type == "image" else MediaType.VIDEO,
                storage_path=storage_path,
                duration_sec=duration_sec,
                resolution=resolution,
                status=MediaStatus.PENDING,
                item_metadata=metadata or {},
            )
            session.add(media_item)
            session.flush()
            session.refresh(media_item)
            session.expunge(media_item)
            logger.info(f"Ingested new MediaItem {media_item.id} (status: pending).")
            return media_item

    def ingest_candidate(self, candidate: Candidate) -> MediaItem:
        """Ingests a standard Candidate object."""
        return self.ingest(
            source_url=candidate.source_url,
            provider=candidate.provider,
            source_id=candidate.source_id,
            metadata=candidate.metadata,
            license_type=candidate.license_type,
            media_type=candidate.media_type,
            duration_sec=candidate.duration_sec,
            resolution=candidate.resolution,
        )

    def search_and_ingest(
        self,
        keyword: str,
        provider: str = "pixabay",
        max_results: int = 10,
        media_type: str = "video",
    ) -> list[MediaItem]:
        """Search a provider and ingest newly discovered media candidates."""
        adapter = self.adapters.get(provider)
        if not adapter:
            raise ValueError(
                f"Unknown provider '{provider}'. Available: {list(self.adapters.keys())}"
            )

        if provider == "pexels" and media_type == "image" and hasattr(adapter, "search_photos"):
            candidates = adapter.search_photos(keyword=keyword, max_results=max_results)
        else:
            candidates = adapter.search(keyword=keyword, max_results=max_results)

        results: list[MediaItem] = []
        for cand in candidates:
            item = self.ingest_candidate(cand)
            results.append(item)
        return results

    def ingest_url_list(
        self,
        urls: list[str],
        provider: str = "manual",
    ) -> list[MediaItem]:
        """Ingest a list of direct URLs."""
        results: list[MediaItem] = []
        for url in urls:
            cand = DirectURLAdapter.url_to_candidate(url, provider=provider)
            item = self.ingest_candidate(cand)
            results.append(item)
        return results

    def ingest_from_provider(
        self,
        provider: URLProvider,
        provider_name: str = "external_db",
    ) -> list[MediaItem]:
        """Ingest from an external DB / URLProvider stream."""
        results: list[MediaItem] = []
        for url, metadata in provider.fetch_urls():
            cand = DirectURLAdapter.url_to_candidate(
                url=url,
                provider=provider_name,
                source_id=metadata.get("id"),
                metadata=metadata,
            )
            item = self.ingest_candidate(cand)
            results.append(item)
        return results


# Module-level convenience functions matching SPEC.md API
_default_orchestrator: Orchestrator | None = None


def get_orchestrator() -> Orchestrator:
    global _default_orchestrator
    if _default_orchestrator is None:
        _default_orchestrator = Orchestrator()
    return _default_orchestrator


def ingest(
    source_url: str,
    provider: str = "manual",
    source_id: str | None = None,
    metadata: dict | None = None,
    license_type: str = "unknown",
    media_type: str = "video",
    duration_sec: float | None = None,
    resolution: str | None = None,
) -> MediaItem:
    return get_orchestrator().ingest(
        source_url=source_url,
        provider=provider,
        source_id=source_id,
        metadata=metadata,
        license_type=license_type,
        media_type=media_type,
        duration_sec=duration_sec,
        resolution=resolution,
    )


def search_and_ingest(
    keyword: str,
    provider: Literal["pixabay", "pexels", "coverr"] = "pixabay",
    max_results: int = 20,
    media_type: str = "video",
) -> list[MediaItem]:
    return get_orchestrator().search_and_ingest(
        keyword=keyword,
        provider=provider,
        max_results=max_results,
        media_type=media_type,
    )


def ingest_url_list(
    urls: list[str],
    provider: str = "manual",
) -> list[MediaItem]:
    return get_orchestrator().ingest_url_list(urls=urls, provider=provider)


def ingest_from_provider(
    provider: URLProvider,
    provider_name: str = "external_db",
) -> list[MediaItem]:
    return get_orchestrator().ingest_from_provider(
        provider=provider,
        provider_name=provider_name,
    )
