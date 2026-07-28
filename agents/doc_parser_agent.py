"""Document parsing agent for PDF, images, tables, text, and Markdown."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import settings


class DocType(str, Enum):
    PDF = "pdf"
    IMAGE = "image"
    TABLE = "table"
    TEXT = "text"
    MARKDOWN = "markdown"
    UNKNOWN = "unknown"


@dataclass
class DocumentChunk:
    """A document chunk with content and metadata."""

    content: str
    doc_id: str
    chunk_index: int
    doc_type: DocType
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None

    @property
    def chunk_id(self) -> str:
        return f"{self.doc_id}#chunk-{self.chunk_index}"


class DocParserAgent:
    """Parse documents, clean text, split into chunks, and attach metadata."""

    SUPPORTED_EXTENSIONS: dict[str, DocType] = {
        ".pdf": DocType.PDF,
        ".png": DocType.IMAGE,
        ".jpg": DocType.IMAGE,
        ".jpeg": DocType.IMAGE,
        ".csv": DocType.TABLE,
        ".xlsx": DocType.TABLE,
        ".xls": DocType.TABLE,
        ".txt": DocType.TEXT,
        ".md": DocType.MARKDOWN,
    }

    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 80

    def __init__(self) -> None:
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=0,
        )

    async def parse(self, file_path: str) -> list[DocumentChunk]:
        doc_type = self._classify(file_path)
        doc_id = self._make_doc_id(file_path)

        if doc_type == DocType.PDF:
            raw_texts = await self._parse_pdf(file_path)
        elif doc_type == DocType.IMAGE:
            raw_texts = await self._parse_image(file_path)
        elif doc_type == DocType.TABLE:
            raw_texts = await self._parse_table(file_path)
        else:
            raw_texts = self._parse_text(file_path)

        cleaned_texts = [self._clean_text(text) for text in raw_texts]
        return self._chunk_texts(cleaned_texts, doc_id, doc_type, file_path)

    async def parse_batch(self, file_paths: list[str]) -> list[DocumentChunk]:
        all_chunks: list[DocumentChunk] = []
        for fp in file_paths:
            all_chunks.extend(await self.parse(fp))
        return all_chunks

    def _classify(self, file_path: str) -> DocType:
        ext = os.path.splitext(file_path)[1].lower()
        return self.SUPPORTED_EXTENSIONS.get(ext, DocType.UNKNOWN)

    @staticmethod
    def _make_doc_id(file_path: str) -> str:
        return hashlib.sha256(file_path.encode()).hexdigest()[:16]

    async def _parse_pdf(self, file_path: str) -> list[str]:
        texts: list[str] = []
        try:
            from PyPDF2 import PdfReader

            reader = PdfReader(file_path)
            for page in reader.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    texts.append(page_text.strip())
        except Exception:
            texts.append(f"[PDF parse failed] {file_path}")

        if not texts:
            texts = await self._pdf_vision_fallback(file_path)

        return texts

    async def _pdf_vision_fallback(self, file_path: str) -> list[str]:
        try:
            from pdf2image import convert_from_path

            images = convert_from_path(file_path, dpi=150, first_page=1, last_page=5)
            texts: list[str] = []
            for img in images:
                texts.append(await self._describe_image_with_llm(img))
            return texts
        except Exception:
            return [f"[PDF vision parse failed] {file_path}"]

    async def _parse_image(self, file_path: str) -> list[str]:
        texts: list[str] = []
        ocr_text = self._ocr(file_path)
        if ocr_text.strip():
            texts.append(ocr_text)

        from PIL import Image

        img = Image.open(file_path)
        texts.append(await self._describe_image_with_llm(img))
        return texts

    @staticmethod
    def _ocr(file_path: str) -> str:
        try:
            import pytesseract
            from PIL import Image

            return pytesseract.image_to_string(Image.open(file_path), lang="chi_sim+eng")
        except Exception:
            return ""

    async def _describe_image_with_llm(self, image: Any) -> str:
        import base64
        import io

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        messages = [
            SystemMessage(content="You are a professional document analysis assistant. Describe all visible text, tables, charts, and layout information."),
            HumanMessage(
                content=[
                    {"type": "text", "text": "Describe all content in this image."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ]
            ),
        ]
        resp = await self.llm.ainvoke(messages)
        return str(resp.content)

    async def _parse_table(self, file_path: str) -> list[str]:
        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext == ".csv":
                return self._parse_csv(file_path)
            return self._parse_excel(file_path)
        except Exception:
            return [f"[Table parse failed] {file_path}"]

    @staticmethod
    def _parse_csv(file_path: str) -> list[str]:
        import csv

        texts: list[str] = []
        with open(file_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            rows: list[str] = []
            for row in reader:
                rows.append(" | ".join(f"{h}: {row.get(h, '')}" for h in headers))
            for i in range(0, len(rows), 20):
                batch = rows[i : i + 20]
                texts.append(f"Headers: {' | '.join(headers)}\n" + "\n".join(batch))
        return texts or ["[Empty CSV]"]

    @staticmethod
    def _parse_excel(file_path: str) -> list[str]:
        try:
            import openpyxl

            wb = openpyxl.load_workbook(file_path, read_only=True)
            texts: list[str] = []
            for sheet in wb.worksheets:
                rows = list(sheet.iter_rows(values_only=True))
                if not rows:
                    continue
                headers = [str(c) if c else "" for c in rows[0]]
                data_rows: list[str] = []
                for row in rows[1:]:
                    data_rows.append(
                        " | ".join(
                            f"{headers[j]}: {row[j]}" if j < len(headers) else str(row[j])
                            for j in range(len(row))
                        )
                    )
                for i in range(0, len(data_rows), 20):
                    batch = data_rows[i : i + 20]
                    texts.append(f"Sheet: {sheet.title}\nHeaders: {' | '.join(headers)}\n" + "\n".join(batch))
            return texts or ["[Empty Excel]"]
        except Exception:
            return [f"[Excel parse failed] {file_path}"]

    @staticmethod
    def _parse_text(file_path: str) -> list[str]:
        with open(file_path, encoding="utf-8") as f:
            return [f.read()]

    @staticmethod
    def _clean_text(text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        lines: list[str] = []
        for line in text.split("\n"):
            line = re.sub(r"[ \t]+", " ", line).strip()
            if line:
                lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def _is_heading_line(line: str) -> tuple[int, str] | None:
        markdown_heading = re.match(r"^(#{1,6})\s+(.+)$", line)
        if markdown_heading:
            return len(markdown_heading.group(1)), markdown_heading.group(2).strip()

        numbered_heading = re.match(r"^(第[一二三四五六七八九十百]+[章节部分]|\d+(?:\.\d+){0,2})[、.\s]+(.+)$", line)
        if numbered_heading:
            return 1, numbered_heading.group(2).strip()

        return None

    @staticmethod
    def _is_table_line(line: str) -> bool:
        return line.count("|") >= 2

    @staticmethod
    def _is_list_line(line: str) -> bool:
        return bool(re.match(r"^(\-|\*|\+|\d+\.)\s+\S+", line))

    @staticmethod
    def _chunk_metadata(
        source: str,
        doc_type: DocType,
        section_title: str,
        parent_section: str,
        char_start: int,
        content: str,
    ) -> dict[str, Any]:
        return {
            "source": source,
            "file_name": os.path.basename(source),
            "section_title": section_title,
            "parent_section": parent_section,
            "doc_type": doc_type.value,
            "char_start": char_start,
            "char_end": char_start + len(content),
        }

    def _append_chunk(
        self,
        chunks: list[DocumentChunk],
        *,
        doc_id: str,
        doc_type: DocType,
        source: str,
        idx: int,
        content: str,
        section_title: str,
        parent_section: str,
        char_start: int,
    ) -> int:
        content = content.strip()
        if not content:
            return idx
        chunks.append(
            DocumentChunk(
                content=content,
                doc_id=doc_id,
                chunk_index=idx,
                doc_type=doc_type,
                metadata=self._chunk_metadata(
                    source,
                    doc_type,
                    section_title,
                    parent_section,
                    char_start,
                    content,
                ),
            )
        )
        return idx + 1

    def _split_long_block(self, block: str) -> list[str]:
        if len(block) <= self.CHUNK_SIZE:
            return [block]

        parts: list[str] = []
        start = 0
        while start < len(block):
            end = min(start + self.CHUNK_SIZE, len(block))
            content = block[start:end].strip()
            if content:
                parts.append(content)
            if end >= len(block):
                break
            start = max(end - self.CHUNK_OVERLAP, start + 1)
        return parts

    def _iter_structured_blocks(self, text: str) -> list[tuple[str, str, str, str]]:
        blocks: list[tuple[str, str, str, str]] = []
        section_stack: list[tuple[int, str]] = []
        buffer: list[str] = []
        buffer_type = "paragraph"
        buffer_section = ""
        buffer_parent = ""

        def flush() -> None:
            nonlocal buffer, buffer_type, buffer_section, buffer_parent
            if buffer:
                blocks.append(("\n".join(buffer).strip(), buffer_type, buffer_section, buffer_parent))
                buffer = []

        for line in text.split("\n"):
            heading = self._is_heading_line(line)
            if heading:
                flush()
                level, title = heading
                while section_stack and section_stack[-1][0] >= level:
                    section_stack.pop()
                parent = section_stack[-1][1] if section_stack else ""
                section_stack.append((level, title))
                blocks.append((line.strip(), "heading", title, parent))
                continue

            current_section = section_stack[-1][1] if section_stack else ""
            current_parent = section_stack[-2][1] if len(section_stack) >= 2 else ""
            line_type = "table" if self._is_table_line(line) else "list" if self._is_list_line(line) else "paragraph"

            if not buffer:
                buffer = [line]
                buffer_type = line_type
                buffer_section = current_section
                buffer_parent = current_parent
                continue

            if line_type != buffer_type:
                flush()
                buffer = [line]
                buffer_type = line_type
                buffer_section = current_section
                buffer_parent = current_parent
                continue

            buffer.append(line)

        flush()
        return blocks

    def _chunk_texts(
        self,
        texts: list[str],
        doc_id: str,
        doc_type: DocType,
        source: str,
    ) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        idx = 0
        for text in texts:
            blocks = self._iter_structured_blocks(text)
            buffer = ""
            char_start = 0
            buffer_section = ""
            buffer_parent = ""

            for block, _block_type, section_title, parent_section in blocks:
                if len(block) > self.CHUNK_SIZE:
                    if buffer:
                        idx = self._append_chunk(
                            chunks,
                            doc_id=doc_id,
                            doc_type=doc_type,
                            source=source,
                            idx=idx,
                            content=buffer,
                            section_title=buffer_section,
                            parent_section=buffer_parent,
                            char_start=char_start,
                        )
                        char_start += max(len(buffer) - self.CHUNK_OVERLAP, 1)
                        buffer = ""
                        buffer_section = ""
                        buffer_parent = ""

                    for part in self._split_long_block(block):
                        idx = self._append_chunk(
                            chunks,
                            doc_id=doc_id,
                            doc_type=doc_type,
                            source=source,
                            idx=idx,
                            content=part,
                            section_title=section_title,
                            parent_section=parent_section,
                            char_start=char_start,
                        )
                        char_start += max(len(part) - self.CHUNK_OVERLAP, 1)
                    continue

                candidate = f"{buffer}\n{block}".strip() if buffer else block
                if len(candidate) <= self.CHUNK_SIZE:
                    buffer = candidate
                    if section_title:
                        buffer_section = section_title
                    if parent_section:
                        buffer_parent = parent_section
                    continue

                if buffer:
                    idx = self._append_chunk(
                        chunks,
                        doc_id=doc_id,
                        doc_type=doc_type,
                        source=source,
                        idx=idx,
                        content=buffer,
                        section_title=buffer_section,
                        parent_section=buffer_parent,
                        char_start=char_start,
                    )
                    overlap = buffer[-self.CHUNK_OVERLAP:].strip()
                    char_start += max(len(buffer) - len(overlap), 1)
                    buffer = f"{overlap}\n{block}".strip() if overlap else block
                    buffer_section = section_title
                    buffer_parent = parent_section
                else:
                    buffer = block
                    buffer_section = section_title
                    buffer_parent = parent_section

            if buffer.strip():
                idx = self._append_chunk(
                    chunks,
                    doc_id=doc_id,
                    doc_type=doc_type,
                    source=source,
                    idx=idx,
                    content=buffer.strip(),
                    section_title=buffer_section,
                    parent_section=buffer_parent,
                    char_start=char_start,
                )
        return chunks
