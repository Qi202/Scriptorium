"""Convert Topic Markdown files into stable memory units."""

import re
from pathlib import Path

from .models import EvidenceAnnotation, MEMORY_ID, MemoryUnit, TopicFormatError
from .syntax import (
    ANY_BLOCK_SUFFIX,
    BLOCK_LINK,
    BLOCK_SUFFIX,
    CITATION_GROUP,
    SINGLE_CITATION,
    definitions,
    definition_match,
    paragraphs,
)


_FOOTNOTE_DEFINITION_LIKE = re.compile(r"^\s*\[\^[^\]\s]+\]:")


def _malformed_footnote_definitions(topics: Path) -> list[str]:
    """List every definition-looking line that violates the Topic contract."""
    malformed = []
    for path in sorted(topics.rglob("*.md")):
        relative = path.relative_to(topics).as_posix()
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if _FOOTNOTE_DEFINITION_LIKE.match(line) and not definition_match(line):
                excerpt = " ".join(line.split())[:180]
                malformed.append(f"{relative}:{line_number}={excerpt!r}")
    return malformed


def _duplicate_footnote_definitions(topics: Path) -> list[str]:
    duplicates = []
    for path in sorted(topics.rglob("*.md")):
        locations: dict[str, list[int]] = {}
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if match := definition_match(line):
                locations.setdefault(match.group("id"), []).append(line_number)
        for citation_id, lines in locations.items():
            if len(lines) > 1:
                duplicates.append(
                    f"id={citation_id} file={path.relative_to(topics).as_posix()} "
                    f"lines={lines!r} occurrences={len(lines)}"
                )
    return duplicates


def _definition_line(lines: list[str], citation_id: str) -> int | None:
    return next(
        (number for number, line in enumerate(lines, 1)
         if (match := definition_match(line)) and match.group("id") == citation_id),
        None,
    )


def _trailing_content_errors(topics: Path) -> list[str]:
    """Find every block paragraph with prose or placeholders after its last citation."""
    errors = []
    for path in sorted(topics.rglob("*.md")):
        relative = path.relative_to(topics).as_posix()
        for paragraph, _headings, line_number in paragraphs(
            path.read_text(encoding="utf-8").splitlines(), with_line_numbers=True,
        ):
            block = BLOCK_SUFFIX.search(paragraph)
            if block is None:
                continue
            body = paragraph[:block.start()].rstrip()
            citations = list(CITATION_GROUP.finditer(body))
            if not citations:
                continue
            trailing = body[citations[-1].end():].strip()
            if trailing:
                excerpt = " ".join(trailing.split())[:120]
                errors.append(
                    f"block={block.group('id')}; file={relative}; line={line_number}; "
                    f"trailing_excerpt={excerpt!r}; trailing_chars={len(trailing)}"
                )
    return errors


def _block_unit(
    paragraph: str,
    headings: tuple[str, ...],
    path: Path,
    topics: Path,
    known_definitions: dict,
    created_order: int,
    line_number: int | None = None,
    *,
    strict: bool,
) -> tuple[list[MemoryUnit], set[str]] | None:
    block = BLOCK_SUFFIX.search(paragraph)
    any_block = ANY_BLOCK_SUFFIX.search(paragraph)
    if any_block and not block:
        raise TopicFormatError(
            f"invalid block_id: {any_block.group('id')} "
            f"file={path.relative_to(topics).as_posix()} line={line_number}"
        )
    if not block:
        return None
    # A merge keeps every absorbed ID on one paragraph. The last is this
    # paragraph's identity; the rest are aliases the caller re-emits so
    # links pointing at them still resolve.
    absorbed = re.findall(r"\^([A-Za-z0-9-]+)", paragraph[block.start():])
    memory_id = absorbed[-1]
    body = paragraph[:block.start()].rstrip()
    evidence = []
    used: set[str] = set()
    cursor = 0
    for group in CITATION_GROUP.finditer(body):
        quote = body[cursor:group.start()].strip()
        ids = [match.group("id") for match in SINGLE_CITATION.finditer(group.group(0))]
        if not quote:
            raise TopicFormatError(
                f"missing memory content before {ids[0]} "
                f"file={path.relative_to(topics).as_posix()} line={line_number}"
            )
        for citation_id in ids:
            if citation_id not in known_definitions:
                raise TopicFormatError(
                    f"undefined footnote: {citation_id} "
                    f"file={path.relative_to(topics).as_posix()} "
                    f"line={line_number} block={memory_id}"
                )
            when, refs, links = known_definitions[citation_id]
            evidence.append(EvidenceAnnotation(
                citation_id=citation_id,
                quote=quote,
                when=when,
                source_refs=refs,
                source_links=links,
            ))
            used.add(citation_id)
        cursor = group.end()
    if strict and not evidence:
        raise TopicFormatError(
            f"memory source footnote required: {memory_id} "
            f"file={path.relative_to(topics).as_posix()} line={line_number}"
        )
    trailing = body[cursor:].strip()
    if strict and trailing:
        relative = path.relative_to(topics).as_posix()
        excerpt = " ".join(trailing.split())[:120]
        raise TopicFormatError(
            f"content after final evidence: {memory_id}; "
            f"file={relative}; line={line_number}; trailing_excerpt={excerpt!r}; "
            f"trailing_chars={len(trailing)}"
        )
    refs: list[str] = []
    links: list[str] = []
    for annotation in evidence:
        for ref, link in zip(annotation.source_refs, annotation.source_links):
            if ref not in refs:
                refs.append(ref)
                links.append(link)
    shared = dict(
        content=SINGLE_CITATION.sub("", body).strip(),
        when=next((item.when for item in evidence if item.when is not None), None),
        source_refs=tuple(refs),
        source_links=tuple(links),
        topic_path=path.relative_to(topics).as_posix(),
        headings=headings,
        created_order=created_order,
        evidence=tuple(evidence),
        relation_targets=tuple(
            match.group("id") for match in BLOCK_LINK.finditer(body)
        ),
    )
    return [
        MemoryUnit(memory_id=value, **shared) for value in absorbed
    ], used


def _legacy_units(
    paragraph: str,
    headings: tuple[str, ...],
    path: Path,
    topics: Path,
    known_definitions: dict,
    created_order: int,
    line_number: int | None = None,
) -> tuple[list[MemoryUnit], set[str]]:
    units = []
    used = set()
    cursor = 0
    for group in CITATION_GROUP.finditer(paragraph):
        content = paragraph[cursor:group.start()].strip()
        ids = [match.group("id") for match in SINGLE_CITATION.finditer(group.group(0))]
        if any(not re.fullmatch(MEMORY_ID, memory_id) for memory_id in ids):
            continue
        if not content:
            raise TopicFormatError(
                f"missing memory content before {ids[0]} "
                f"file={path.relative_to(topics).as_posix()} line={line_number}"
            )
        for memory_id in ids:
            if memory_id not in known_definitions:
                raise TopicFormatError(
                    f"undefined footnote: {memory_id} "
                    f"file={path.relative_to(topics).as_posix()} "
                    f"line={line_number}"
                )
            when, refs, links = known_definitions[memory_id]
            units.append(MemoryUnit(
                memory_id=memory_id,
                content=content,
                when=when,
                source_refs=refs,
                source_links=links,
                topic_path=path.relative_to(topics).as_posix(),
                headings=headings,
                created_order=created_order + len(units),
            ))
            used.add(memory_id)
        cursor = group.end()
    return units, used


def parse_topic_tree(topics: Path, *, strict: bool = True) -> list[MemoryUnit]:
    """Return memory units in their current Markdown occurrence order."""
    topics = Path(topics)
    if strict and (malformed := _malformed_footnote_definitions(topics)):
        raise TopicFormatError(
            "malformed footnote definitions (repair all): " + "; ".join(malformed)
        )
    if strict and (duplicates := _duplicate_footnote_definitions(topics)):
        raise TopicFormatError(
            "duplicate footnote definitions (repair all): "
            + " | ".join(duplicates)
        )
    if strict and (trailing := _trailing_content_errors(topics)):
        raise TopicFormatError(
            "content after final evidence (repair all): " + " | ".join(trailing)
        )
    units: list[MemoryUnit] = []
    seen: set[str] = set()
    seen_locations: dict[str, str] = {}
    duplicate_ids: list[str] = []
    unused_definitions: list[str] = []
    for path in sorted(topics.rglob("*.md")):
        lines = path.read_text(encoding="utf-8").splitlines()
        known_definitions = definitions(
            lines, source_path=path.relative_to(topics).as_posix()
        )
        used_definitions: set[str] = set()
        for paragraph, headings, line_number in paragraphs(lines, with_line_numbers=True):
            parsed = _block_unit(
                paragraph,
                headings,
                path,
                topics,
                known_definitions,
                len(units),
                line_number,
                strict=strict,
            )
            if parsed:
                block_units, used = parsed
                for unit in block_units:
                    if unit.memory_id in seen:
                        duplicate_ids.append(
                            f"id={unit.memory_id} first={seen_locations[unit.memory_id]} "
                            f"duplicate={path.relative_to(topics).as_posix()}:{line_number}"
                        )
                        continue
                    units.append(unit)
                    seen.add(unit.memory_id)
                    seen_locations[unit.memory_id] = (
                        f"{path.relative_to(topics).as_posix()}:{line_number}"
                    )
                used_definitions.update(used)
                continue
            legacy, used = _legacy_units(
                paragraph,
                headings,
                path,
                topics,
                known_definitions,
                len(units),
                line_number,
            )
            if strict and not legacy:
                excerpt = " ".join(paragraph.split())[:180]
                raise TopicFormatError(
                    "memory block ID required: "
                    f"{path.relative_to(topics)}; line={line_number}; "
                    f"paragraph={excerpt!r}"
                )
            for unit in legacy:
                if unit.memory_id in seen:
                    duplicate_ids.append(
                        f"id={unit.memory_id} first={seen_locations[unit.memory_id]} "
                        f"duplicate={path.relative_to(topics).as_posix()}:{line_number}"
                    )
                    continue
                units.append(unit)
                seen.add(unit.memory_id)
                seen_locations[unit.memory_id] = (
                    f"{path.relative_to(topics).as_posix()}:{line_number}"
                )
            used_definitions.update(used)
        unused = set(known_definitions) - used_definitions
        if strict and unused:
            for citation_id in sorted(unused):
                unused_definitions.append(
                    f"id={citation_id} file={path.relative_to(topics).as_posix()} "
                    f"line={_definition_line(lines, citation_id)}"
                )
    if duplicate_ids:
        raise TopicFormatError(
            "duplicate memory_ids (repair all): " + " | ".join(duplicate_ids)
        )
    if strict and unused_definitions:
        raise TopicFormatError(
            "unused footnote definitions (repair all): "
            + " | ".join(unused_definitions)
        )
    return units


def topic_prose(topics: Path) -> str:
    """Return normalized non-structural Topic prose, including unbound text."""
    values = []
    for path in sorted(Path(topics).rglob("*.md")):
        lines = path.read_text(encoding="utf-8").splitlines()
        for paragraph, _headings in paragraphs(lines):
            value = SINGLE_CITATION.sub("", paragraph)
            value = BLOCK_SUFFIX.sub("", value)
            values.append(" ".join(value.split()))
    return "\n".join(values)
