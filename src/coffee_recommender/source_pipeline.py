from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from .catalogue_store import _parse_embedding
from .openai_client import DEFAULT_CHAT_MODEL, DEFAULT_EMBEDDING_MODEL
from .process_data.embed_coffee import embed_coffee_record
from .process_data.extract_sensory import extract_sensory_vector_llm
from .process_data.parse_metadata import parse_metadata_text
from .reviewed_coffee_url import extract_product_text_from_html, fetch_url_html
from .db.models import (
    CatalogueCoffeeModel,
    CoffeeSourceModel,
    EmbeddingModel,
    SensoryProfileModel,
    SourcePageModel,
    SourcePageParseRunModel,
    SourceStoreModel,
)

SCHEMA_VERSION = "coffee_record_v1"
PARSER_VERSION = "page_parser_v2"
SOURCE_TYPE_FETCHED_PAGE = "fetched_page"


@dataclass(frozen=True)
class FetchPageResult:
    source_page_id: int
    normalized_url: str
    content_hash: str


@dataclass(frozen=True)
class ParseStoredPageResult:
    parse_run_id: int
    coffee_id: str
    parse_status: str


def ensure_source_store(
    *,
    session_factory: sessionmaker[Session],
    name: str,
    base_url: str,
    allowed_domains: list[str],
) -> int:
    with session_factory() as session:
        existing = session.scalar(
            select(SourceStoreModel).where(SourceStoreModel.base_url == base_url)
        )
        if existing is not None:
            existing.name = name
            existing.allowed_domains_json = allowed_domains
            session.commit()
            return existing.id

        store = SourceStoreModel(
            name=name,
            base_url=base_url,
            allowed_domains_json=allowed_domains,
        )
        session.add(store)
        session.commit()
        return store.id


def fetch_and_store_page(
    *,
    session_factory: sessionmaker[Session],
    url: str,
    store_id: int | None = None,
    page_type: str = "product",
) -> FetchPageResult:
    normalized_url, html = fetch_url_html(url)
    visible_text = extract_product_text_from_html(html)
    content_hash = hashlib.sha256(visible_text.encode("utf-8")).hexdigest()

    with session_factory() as session:
        existing = _get_existing_source_page(session, normalized_url)
        if existing is None:
            existing = SourcePageModel(
                store_id=store_id,
                source_url=normalized_url,
                normalized_url=normalized_url,
                page_type=page_type,
                fetch_status="fetched",
            )
            session.add(existing)

        _apply_source_page_update(
            existing,
            store_id=store_id,
            normalized_url=normalized_url,
            page_type=page_type,
            html=html,
            visible_text=visible_text,
            content_hash=content_hash,
        )

        try:
            session.commit()
        except IntegrityError:
            session.rollback()
            existing = _get_existing_source_page(session, normalized_url)
            if existing is None:
                raise
            _apply_source_page_update(
                existing,
                store_id=store_id,
                normalized_url=normalized_url,
                page_type=page_type,
                html=html,
                visible_text=visible_text,
                content_hash=content_hash,
            )
            session.commit()

        return FetchPageResult(
            source_page_id=existing.id,
            normalized_url=normalized_url,
            content_hash=content_hash,
        )


def parse_stored_page(
    *,
    session_factory: sessionmaker[Session],
    source_page_id: int,
    metadata_model: str = DEFAULT_CHAT_MODEL,
    sensory_model: str = DEFAULT_CHAT_MODEL,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    schema_version: str = SCHEMA_VERSION,
    parser_version: str = PARSER_VERSION,
) -> ParseStoredPageResult:
    with session_factory() as session:
        source_page = session.get(SourcePageModel, source_page_id)
        if source_page is None:
            raise KeyError(f"Source page not found: {source_page_id}")
        if not source_page.visible_text:
            raise ValueError(f"Source page {source_page_id} does not have visible text.")

        coffee = parse_metadata_text(
            source_page.visible_text,
            source=f"source_page:{source_page.id}",
            source_url=source_page.normalized_url,
            model=metadata_model,
        )
        sensory = extract_sensory_vector_llm(coffee, model=sensory_model)
        embedding = embed_coffee_record(coffee, model=embedding_model)

        catalogue = session.get(CatalogueCoffeeModel, coffee.coffee_id)
        if catalogue is None:
            catalogue = CatalogueCoffeeModel(id=coffee.coffee_id, name=coffee.name)
            session.add(catalogue)

        catalogue.name = coffee.name
        catalogue.roaster = coffee.roaster
        catalogue.origin_country = coffee.origin_country
        catalogue.region = coffee.region
        catalogue.producer = coffee.producer
        catalogue.farm = coffee.farm
        catalogue.process_primary = coffee.process.value if hasattr(coffee.process, "value") else str(coffee.process)
        catalogue.process_detail = catalogue.process_primary
        catalogue.variety_text = ", ".join(coffee.variety) if coffee.variety else None
        catalogue.roast_level = (
            coffee.roast_level.value if hasattr(coffee.roast_level, "value") else str(coffee.roast_level)
        )
        catalogue.tasting_notes_json = coffee.tasting_notes
        catalogue.description = coffee.description
        catalogue.price_minor = int(round(coffee.price * 100)) if coffee.price is not None else None
        catalogue.currency = coffee.currency
        catalogue.weight_g = coffee.weight_g
        catalogue.is_active = True

        source_record = session.scalar(
            select(CoffeeSourceModel)
            .where(CoffeeSourceModel.coffee_id == coffee.coffee_id)
            .where(CoffeeSourceModel.source_url == source_page.normalized_url)
        )
        if source_record is None:
            source_record = CoffeeSourceModel(
                coffee_id=coffee.coffee_id,
                source_type=SOURCE_TYPE_FETCHED_PAGE,
                source_url=source_page.normalized_url,
                raw_text=source_page.visible_text,
            )
            session.add(source_record)

        source_record.raw_text = source_page.visible_text
        source_record.scraped_at = source_page.fetched_at
        source_record.parser_version = parser_version
        source_record.extraction_model = metadata_model
        source_record.extraction_status = "parsed"

        for existing in session.scalars(
            select(SensoryProfileModel).where(SensoryProfileModel.coffee_id == coffee.coffee_id)
        ).all():
            session.delete(existing)
        session.add(
            SensoryProfileModel(
                coffee_id=coffee.coffee_id,
                profile_version=schema_version,
                acidity=Decimal(str(sensory.acidity)),
                sweetness=Decimal(str(sensory.sweetness)),
                body=Decimal(str(sensory.body)),
                bitterness=Decimal(str(sensory.bitterness)),
                fruitiness=Decimal(str(sensory.fruitiness)),
                chocolate_nutty=Decimal(str(sensory.chocolate_nutty)),
                floral=Decimal(str(sensory.floral)),
                funky_fermented=Decimal(str(sensory.funky_fermented)),
                roasty=Decimal(str(sensory.roasty)),
                clean_cup=Decimal(str(sensory.clean_cup)),
                confidence=Decimal(str(sensory.confidence)),
                evidence_json=sensory.evidence,
                model_name=sensory_model,
            )
        )

        for existing in session.scalars(
            select(EmbeddingModel).where(EmbeddingModel.coffee_id == coffee.coffee_id)
        ).all():
            session.delete(existing)
        session.add(
            EmbeddingModel(
                coffee_id=coffee.coffee_id,
                embedding_model=embedding_model,
                vector_json=_parse_embedding(embedding),
            )
        )

        parse_run = SourcePageParseRunModel(
            source_page_id=source_page.id,
            coffee_id=coffee.coffee_id,
            schema_version=schema_version,
            parser_version=parser_version,
            extraction_model=metadata_model,
            parse_status="parsed",
            warnings_json=[],
        )
        session.add(parse_run)
        session.commit()

        return ParseStoredPageResult(
            parse_run_id=parse_run.id,
            coffee_id=coffee.coffee_id,
            parse_status=parse_run.parse_status,
        )


def _get_existing_source_page(session: Session, normalized_url: str) -> SourcePageModel | None:
    return session.scalar(
        select(SourcePageModel).where(
            or_(
                SourcePageModel.normalized_url == normalized_url,
                SourcePageModel.source_url == normalized_url,
            )
        )
    )


def _apply_source_page_update(
    source_page: SourcePageModel,
    *,
    store_id: int | None,
    normalized_url: str,
    page_type: str,
    html: str,
    visible_text: str,
    content_hash: str,
) -> None:
    source_page.store_id = store_id
    source_page.source_url = normalized_url
    source_page.normalized_url = normalized_url
    source_page.page_type = page_type
    source_page.fetch_status = "fetched"
    source_page.html_content = html
    source_page.visible_text = visible_text
    source_page.content_hash = content_hash
    source_page.fetched_at = datetime.now(UTC)
    source_page.last_error = None
