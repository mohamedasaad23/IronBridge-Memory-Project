"""
rag/chunking.py

Splits each policy document in rag/corpus/*.md into section-level chunks.

Convention match with memory/ and context_eval/:
- Pure stdlib, no network calls, fully deterministic.
- Each chunk carries enough metadata (doc_id, topic, section_id, heading,
  last_reviewed) to support both citation-heavy queries ("what does section
  4.2b say") and topic pre-filtering in the vector store.

Chunk boundary rule: a new chunk starts at every "### <section_id> <heading>"
markdown header. YAML frontmatter (--- ... ---) at the top of the file
supplies doc-level metadata (topic, doc_id, last_reviewed).
"""

from __future__ import annotations

import re
import glob
import os
from dataclasses import dataclass, field
from typing import List

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "corpus")

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
SECTION_RE = re.compile(r"^###\s+(\S+)\s+(.+)$", re.MULTILINE)


@dataclass
class Chunk:
    chunk_id: str          # f"{doc_id}:{section_id}"
    doc_id: str
    topic: str
    section_id: str
    heading: str
    last_reviewed: str
    text: str               # heading + body, used for embedding/BM25
    source_file: str
    field_order: int = field(default=0)  # position within the doc, for stable sort


def _parse_frontmatter(raw: str) -> dict:
    m = FRONTMATTER_RE.match(raw)
    if not m:
        return {}
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta


def chunk_document(path: str) -> List[Chunk]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    meta = _parse_frontmatter(raw)
    doc_id = meta.get("doc_id", os.path.basename(path))
    topic = meta.get("topic", "unknown")
    last_reviewed = meta.get("last_reviewed", "unknown")

    body = FRONTMATTER_RE.sub("", raw, count=1)

    matches = list(SECTION_RE.finditer(body))
    chunks: List[Chunk] = []
    for i, m in enumerate(matches):
        section_id, heading = m.group(1), m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        section_text = body[start:end].strip()
        full_text = f"{heading}\n{section_text}"
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}:{section_id}",
                doc_id=doc_id,
                topic=topic,
                section_id=section_id,
                heading=heading,
                last_reviewed=last_reviewed,
                text=full_text,
                source_file=os.path.basename(path),
                field_order=i,
            )
        )
    return chunks


def chunk_corpus(corpus_dir: str = CORPUS_DIR) -> List[Chunk]:
    all_chunks: List[Chunk] = []
    for path in sorted(glob.glob(os.path.join(corpus_dir, "*.md"))):
        all_chunks.extend(chunk_document(path))
    return all_chunks


if __name__ == "__main__":
    chunks = chunk_corpus()
    print(f"Chunked {len(chunks)} sections from {CORPUS_DIR}/*.md")
    for c in chunks:
        print(f"  {c.chunk_id:28s} topic={c.topic:20s} heading={c.heading}")
