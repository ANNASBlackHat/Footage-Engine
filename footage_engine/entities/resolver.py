"""Entity resolution and management."""

import logging
from typing import Optional
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from footage_engine.config import get_settings
from footage_engine.models.db import get_db_session
from footage_engine.models.media import Entity

logger = logging.getLogger(__name__)


class EntityResolver:
    """Manages canonical entity resolution and aliases."""

    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or get_settings().DATABASE_URL

    def _find_entity_in_session(self, session: Session, name_or_alias: str) -> Optional[Entity]:
        """Looks up an entity by name (case-insensitive) or by matching inside its aliases."""
        clean_target = name_or_alias.strip().lower()
        if not clean_target:
            return None

        # 1. Exact match on canonical name (case-insensitive)
        stmt = select(Entity).where(func.lower(Entity.name) == clean_target)
        entity = session.execute(stmt).scalars().first()
        if entity:
            return entity

        # 2. Match within aliases array.
        # Since aliases is stored as a JSON array (portable across SQLite and Postgres),
        # we can query all entities or iterate to find a case-insensitive match.
        all_entities = session.execute(select(Entity)).scalars().all()
        for ent in all_entities:
            if ent.aliases:
                for alias in ent.aliases:
                    if alias.strip().lower() == clean_target:
                        return ent

        return None

    def resolve(
        self,
        name_or_alias: str,
        session: Optional[Session] = None,
    ) -> Optional[Entity]:
        """Resolves an entity by name or alias without creating a new one."""
        if not name_or_alias or not name_or_alias.strip():
            return None

        if session is not None:
            return self._find_entity_in_session(session, name_or_alias)

        with get_db_session(self.database_url) as sess:
            ent = self._find_entity_in_session(sess, name_or_alias)
            if ent:
                sess.expunge(ent)
            return ent

    def resolve_or_create(
        self,
        name: str,
        entity_type: str = "other",
        aliases: Optional[list[str]] = None,
        notes: Optional[str] = None,
        session: Optional[Session] = None,
    ) -> Entity:
        """Resolves an existing entity by name or alias, or creates a new canonical entity."""
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("Entity name cannot be empty.")

        clean_aliases = [a.strip() for a in (aliases or []) if a.strip()]

        if session is not None:
            existing = self._find_entity_in_session(session, clean_name)
            if existing:
                # Merge any new aliases if provided
                if clean_aliases:
                    current = set(existing.aliases or [])
                    new_set = current.union(clean_aliases)
                    if len(new_set) > len(current):
                        existing.aliases = list(new_set)
                        session.flush()
                return existing

            # Create new entity in existing session
            new_ent = Entity(
                name=clean_name,
                entity_type=entity_type or "other",
                aliases=clean_aliases,
                notes=notes,
            )
            session.add(new_ent)
            session.flush()
            logger.info(f"Created new Entity: {new_ent.name} (id={new_ent.id}, type={new_ent.entity_type})")
            return new_ent

        # Manage our own session
        with get_db_session(self.database_url) as sess:
            existing = self._find_entity_in_session(sess, clean_name)
            if existing:
                if clean_aliases:
                    current = set(existing.aliases or [])
                    new_set = current.union(clean_aliases)
                    if len(new_set) > len(current):
                        existing.aliases = list(new_set)
                        sess.flush()
                sess.expunge(existing)
                return existing

            new_ent = Entity(
                name=clean_name,
                entity_type=entity_type or "other",
                aliases=clean_aliases,
                notes=notes,
            )
            sess.add(new_ent)
            sess.flush()
            sess.refresh(new_ent)
            sess.expunge(new_ent)
            logger.info(f"Created new Entity: {new_ent.name} (id={new_ent.id}, type={new_ent.entity_type})")
            return new_ent

    def get_entity(self, entity_id: str, session: Optional[Session] = None) -> Optional[Entity]:
        """Fetches an entity by primary key ID."""
        if not entity_id:
            return None

        if session is not None:
            return session.get(Entity, entity_id)

        with get_db_session(self.database_url) as sess:
            ent = sess.get(Entity, entity_id)
            if ent:
                sess.expunge(ent)
            return ent

    def list_entities(
        self,
        entity_type: Optional[str] = None,
        session: Optional[Session] = None,
    ) -> list[Entity]:
        """Lists all registered entities, optionally filtered by type."""
        stmt = select(Entity)
        if entity_type:
            stmt = stmt.where(Entity.entity_type == entity_type)
        stmt = stmt.order_by(Entity.name.asc())

        if session is not None:
            return list(session.execute(stmt).scalars().all())

        with get_db_session(self.database_url) as sess:
            entities = list(sess.execute(stmt).scalars().all())
            for ent in entities:
                sess.expunge(ent)
            return entities

    def add_aliases(
        self,
        entity_id: str,
        new_aliases: list[str],
        session: Optional[Session] = None,
    ) -> Optional[Entity]:
        """Appends new aliases to an existing entity."""
        clean = [a.strip() for a in new_aliases if a.strip()]
        if not clean:
            return self.get_entity(entity_id, session=session)

        def _update(sess: Session) -> Optional[Entity]:
            ent = sess.get(Entity, entity_id)
            if not ent:
                return None
            current = set(ent.aliases or [])
            ent.aliases = list(current.union(clean))
            sess.flush()
            return ent

        if session is not None:
            return _update(session)

        with get_db_session(self.database_url) as sess:
            ent = _update(sess)
            if ent:
                sess.expunge(ent)
            return ent
