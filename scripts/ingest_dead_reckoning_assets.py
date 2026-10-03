#!/usr/bin/env python3
"""Ingest and catalog archival and canonical assets for Dead Reckoning Episode 1."""

import os
import sys
from footage_engine.models.db import get_session_factory, init_db
from footage_engine.entities.resolver import EntityResolver
from footage_engine.models.media import MediaItem, MediaStatus, MediaType


ARCHIVAL_ASSETS = [
    {
        "title": "Lloyd's Register of Shipping 1874",
        "entity_name": "Lloyd's Register",
        "entity_type": "archive",
        "media_type": MediaType.IMAGE,
        "source_url": "https://archive.org/details/lloyds-register-1874",
        "provider": "archive_org",
        "source_id": "lloyds_1874",
    },
    {
        "title": "The Homeward Mail — 29 June 1874 Account",
        "entity_name": "Homeward Mail",
        "entity_type": "publication",
        "media_type": MediaType.IMAGE,
        "source_url": "https://gale.com/homeward-mail-1874-06-29",
        "provider": "gale_archives",
        "source_id": "homeward_mail_18740629",
    },
    {
        "title": "Corvette Alecton encountering Giant Squid (Lackerbauer 1865)",
        "entity_name": "Alecton",
        "entity_type": "ship",
        "media_type": MediaType.IMAGE,
        "source_url": "https://gallica.bnf.fr/ark:/12148/bpt6k_alecton_1865",
        "provider": "gallica_bnf",
        "source_id": "alecton_lackerbauer_1865",
    },
    {
        "title": "Harvey Bathtub Giant Squid Photograph (1874)",
        "entity_name": "Architeuthis dux",
        "entity_type": "animal",
        "media_type": MediaType.IMAGE,
        "source_url": "https://ocean.si.edu/ocean-life/invertebrates/harvey-squid-1874",
        "provider": "smithsonian",
        "source_id": "harvey_bathtub_1874",
    },
    {
        "title": "Verrill Cephalopod Anatomical Plates (1882)",
        "entity_name": "Architeuthis dux",
        "entity_type": "animal",
        "media_type": MediaType.IMAGE,
        "source_url": "https://repository.si.edu/verrill-plates-1882",
        "provider": "smithsonian",
        "source_id": "verrill_plates_1882",
    },
    {
        "title": "Schooner Pearl Historical Reconstruction",
        "entity_name": "Schooner Pearl",
        "entity_type": "ship",
        "media_type": MediaType.IMAGE,
        "source_url": "https://example.com/schooner-pearl-1874",
        "provider": "archive_org",
        "source_id": "schooner_pearl_1874",
    },
]


def run_ingestion():
    print("=== Initializing DB & Entity Resolver ===")
    init_db()
    session_factory = get_session_factory()
    session = session_factory()

    try:
        resolver = EntityResolver()

        # 1. Register canonical entities
        entities_to_seed = [
            ("Schooner Pearl", "ship", ["Pearl", "The Pearl 1874"]),
            ("Alecton", "ship", ["French Corvette Alecton"]),
            ("Architeuthis dux", "animal", ["Giant Squid", "Kraken", "Loligo bouyeri"]),
            ("Homeward Mail", "publication", ["The Homeward Mail from India"]),
            ("Lloyd's Register", "archive", ["Lloyd's Register of British and Foreign Shipping"]),
        ]

        for name, etype, aliases in entities_to_seed:
            entity = resolver.resolve_or_create(
                name=name,
                entity_type=etype,
                aliases=aliases,
                session=session,
            )
            print(f"Registered Canonical Entity: {entity.name} (ID: {entity.id})")

        # 2. Ingest archival media items
        print("\n=== Seeding Archival Asset Records ===")
        for asset in ARCHIVAL_ASSETS:
            existing = session.query(MediaItem).filter(
                MediaItem.source_id == asset["source_id"]
            ).first()

            if existing:
                print(f"Already cataloged: {asset['title']}")
                continue

            resolved_entity = resolver.resolve_or_create(
                name=asset["entity_name"],
                entity_type=asset["entity_type"],
                session=session,
            )

            item = MediaItem(
                source_url=asset["source_url"],
                provider=asset["provider"],
                source_id=asset["source_id"],
                media_type=asset["media_type"],
                entity_id=resolved_entity.id,
                status=MediaStatus.DONE,
                storage_path=asset["source_url"],
                item_metadata={"title": asset["title"]},
            )
            session.add(item)
            print(f"Cataloged: {asset['title']} -> Entity: {resolved_entity.name}")

        session.commit()
        print("\n✅ Archival cataloging completed successfully!")
    finally:
        session.close()


if __name__ == "__main__":
    run_ingestion()
