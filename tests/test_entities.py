"""Tests for Entity data model and EntityResolver."""

import pytest
from sqlalchemy.exc import IntegrityError

from footage_engine.entities import EntityResolver
from footage_engine.models.db import get_db_session, init_db
from footage_engine.models.media import Entity, MediaItem, MediaStatus


def test_entity_creation_and_uniqueness(test_settings):
    init_db(test_settings.DATABASE_URL)
    with get_db_session(test_settings.DATABASE_URL) as session:
        ent = Entity(
            name="USS Cyclops",
            aliases=["Cyclops", "AC-4", "the collier"],
            entity_type="ship",
            notes="Collier ship that disappeared in Bermuda Triangle",
        )
        session.add(ent)
        session.flush()

        assert ent.id is not None
        assert len(ent.id) == 36
        assert ent.name == "USS Cyclops"
        assert len(ent.aliases) == 3
        assert ent.entity_type == "ship"

    # Verify unique constraint on name
    with pytest.raises(IntegrityError):
        with get_db_session(test_settings.DATABASE_URL) as session:
            duplicate = Entity(
                name="USS Cyclops",
                entity_type="ship",
            )
            session.add(duplicate)
            session.flush()


def test_entity_resolver_resolve_and_create(test_settings):
    init_db(test_settings.DATABASE_URL)
    resolver = EntityResolver(database_url=test_settings.DATABASE_URL)

    # 1. Create a new entity
    ent1 = resolver.resolve_or_create(
        name="Aye-aye",
        entity_type="animal",
        aliases=["aye aye lemur", "Daubentonia madagascariensis"],
    )
    assert ent1.id is not None
    assert ent1.name == "Aye-aye"
    assert ent1.entity_type == "animal"
    assert len(ent1.aliases) == 2

    # 2. Exact match resolve (case-insensitive)
    ent2 = resolver.resolve_or_create(name="aye-aye")
    assert ent2.id == ent1.id

    # 3. Match by alias
    ent3 = resolver.resolve_or_create(name="aye aye lemur")
    assert ent3.id == ent1.id

    # 4. Resolve another alias without creating
    ent4 = resolver.resolve("Daubentonia madagascariensis")
    assert ent4 is not None
    assert ent4.id == ent1.id

    # 5. Add alias to existing entity
    updated = resolver.add_aliases(ent1.id, ["long-fingered lemur"])
    assert updated is not None
    assert "long-fingered lemur" in updated.aliases

    # 6. List entities
    all_ents = resolver.list_entities()
    assert len(all_ents) == 1
    assert all_ents[0].name == "Aye-aye"


def test_entity_media_item_linking(test_settings):
    init_db(test_settings.DATABASE_URL)
    resolver = EntityResolver(database_url=test_settings.DATABASE_URL)
    cyclops = resolver.resolve_or_create(name="USS Cyclops", entity_type="ship")

    with get_db_session(test_settings.DATABASE_URL) as session:
        item = MediaItem(
            entity_id=cyclops.id,
            provider="manual",
            source_url="https://example.com/cyclops_dock.mp4",
            storage_path="cyclops_dock.mp4",
            duration_sec=25.0,
        )
        session.add(item)
        session.flush()

        assert item.entity_id == cyclops.id
        assert item.entity.name == "USS Cyclops"


def test_orchestrator_ingest_with_entity(test_orchestrator):
    # Ingest footage specifying entity_name
    item = test_orchestrator.ingest(
        source_url="https://example.com/cyclops_voyage.mp4",
        provider="manual",
        entity_name="USS Cyclops",
        entity_type="ship",
    )

    assert item.entity_id is not None
    ent = test_orchestrator.entity_resolver.get_entity(item.entity_id)
    assert ent is not None
    assert ent.name == "USS Cyclops"

    # Ingest second footage with alias of the same entity
    test_orchestrator.entity_resolver.add_aliases(ent.id, ["Cyclops Collier"])
    item2 = test_orchestrator.ingest(
        source_url="https://example.com/cyclops_port.mp4",
        provider="manual",
        entity_name="Cyclops Collier",
    )
    assert item2.entity_id == ent.id

    # Ingest footage without entity (backward compatibility)
    item_unassigned = test_orchestrator.ingest(
        source_url="https://example.com/generic_ocean.mp4",
        provider="manual",
    )
    assert item_unassigned.entity_id is None
