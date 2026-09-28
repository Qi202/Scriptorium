"""Low-level parsing of headings, paragraphs, footnotes, and block links."""

import re
from collections.abc import Iterable, Iterator

from .models import (
    BLOCK_ID,
    FOOTNOTE_ID,
    TEMPORAL_VALUE_PATTERN,
    TopicFormatError,
    is_valid_temporal_value,
)


CITATION_GROUP = re.compile(rf"(?:\[\^(?P<id>{FOOTNOTE_ID})\])+")
SINGLE_CITATION = re.compile(rf"\[\^(?P<id>{FOOTNOTE_ID})\]")
DEFINITION = re.compile(
    rf"^\[\^(?P<id>{FOOTNOTE_ID})\]:\s*"
    rf"Time:\s*(?P<tick>`)?(?P<when>{TEMPORAL_VALUE_PATTERN}|undated)(?(tick)`)"
    r"\s*;\s*Sources:\s*(?P<sources>.+?)\s*$"
)
LEGACY_DEFINITION = re.compile(
    rf"^\[\^(?P<id>{FOOTNOTE_ID})\]:\s*"
    rf"(?P<when>{TEMPORAL_VALUE_PATTERN}|undated)"
    r"\s*·\s*Sources:\s*(?P<sources>.+?)\s*$"
)
LINK = re.compile(r"\[([^]]+)\]\(([^)]+)\)")
# A merged paragraph carries every ID it absorbed, so the suffix is a run.
BLOCK_SUFFIX = re.compile(rf"(?:\s+\^{BLOCK_ID})*\s+\^(?P<id>{BLOCK_ID})\s*$")
BLOCK_SUFFIX_RUN = re.compile(rf"(?:\s+\^{BLOCK_ID})+\s*$")
ANY_BLOCK_SUFFIX = re.compile(r"\s+\^(?P<id>\S+)\s*$")
BLOCK_LINK = re.compile(rf"\[[^]]+\]\([^)#]*#\^(?P<id>{BLOCK_ID})\)")
SOURCE_HANDLE = re.compile(
    r"^(D\d+:\d+|[^/\s,]+/[^/\s,]+/[^/\s,]+)(?:\s*(?:,|·)\s*|\s+|$)"
)


def definition_match(line: str) -> re.Match[str] | None:
    """Match canonical definitions plus legacy definitions for migration."""
    return DEFINITION.match(line) or LEGACY_DEFINITION.match(line)


def render_definition(
    citation_id: str,
    when: str | None,
    sources: Iterable[str],
) -> str:
    return (
        f"[^{citation_id}]: Time: `{when or 'undated'}`; Sources: "
        + ", ".join(sources)
    )


def source_reference(label: str, target: str) -> str:
    handle = SOURCE_HANDLE.match(label.strip())
    if handle:
        return handle.group(1)
    anchor = re.search(r"#d(\d+)-(\d+)$", target, re.IGNORECASE)
    return f"D{anchor.group(1)}:{anchor.group(2)}" if anchor else label


def definitions(
    lines: list[str],
    *, source_path: str | None = None,
) -> dict[str, tuple[str | None, tuple[str, ...], tuple[str, ...]]]:
    values = {}
    for line_number, line in enumerate(lines, 1):
        match = definition_match(line)
        if not match:
            continue
        citation_id = match.group("id")
        location = (
            f" file={source_path} line={line_number}"
            if source_path is not None else ""
        )
        if citation_id in values:
            raise TopicFormatError(
                f"duplicate footnote definition: {citation_id}{location}"
            )
        when = match.group("when")
        if when != "undated" and not is_valid_temporal_value(when):
            raise TopicFormatError(f"invalid evidence date: {when}{location} footnote={citation_id}")
        links = LINK.findall(match.group("sources"))
        if not links:
            raise TopicFormatError(
                f"memory source links required: {citation_id}{location}"
            )
        values[citation_id] = (
            None if when == "undated" else when,
            tuple(source_reference(label, target) for label, target in links),
            tuple(target for _label, target in links),
        )
    return values


def paragraphs(
    lines: list[str], *, with_line_numbers: bool = False,
) -> Iterator[tuple[str, tuple[str, ...]] | tuple[str, tuple[str, ...], int]]:
    headings: list[str] = []
    paragraph: list[str] = []
    paragraph_headings: tuple[str, ...] = ()
    paragraph_start = 0
    in_fence = False

    def flush() -> tuple[str, tuple[str, ...]] | tuple[str, tuple[str, ...], int] | None:
        nonlocal paragraph
        if not paragraph:
            return None
        value = "\n".join(paragraph).strip()
        paragraph = []
        if not value:
            return None
        if with_line_numbers:
            return value, paragraph_headings, paragraph_start
        return value, paragraph_headings

    for line_number, line in enumerate(lines + [""], start=1):
        if line.lstrip().startswith("```"):
            if result := flush():
                yield result
            in_fence = not in_fence
            continue
        if in_fence or definition_match(line) or re.match(
            r"\s*(?:<!--|<a\s)", line
        ):
            if result := flush():
                yield result
            continue
        if heading := re.match(r"^(#{1,6})\s+(.+?)\s*$", line):
            if result := flush():
                yield result
            level = len(heading.group(1))
            headings = headings[: level - 1] + [heading.group(2)]
            continue
        if not line.strip():
            if result := flush():
                yield result
            continue
        if not paragraph:
            paragraph_headings = tuple(headings)
            paragraph_start = line_number
        paragraph.append(line)
