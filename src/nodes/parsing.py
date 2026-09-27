import logging
import tempfile
from pathlib import Path

import fitz
import requests

from src.state import AgentState


logger = logging.getLogger(__name__)

_DOWNLOAD_TIMEOUT_SECONDS = 30
_MAX_PDF_BYTES = 50 * 1024 * 1024
_MAX_PAGES = 60
_MIN_TEXT_CHARS = 500


def _download_pdf(url: str) -> bytes:
    """Download a PDF with one retry."""

    last_error = None

    for attempt in range(2):
        try:
            response = requests.get(
                url,
                timeout=_DOWNLOAD_TIMEOUT_SECONDS,
                stream=True,
            )

            response.raise_for_status()

            content = bytearray()

            for chunk in response.iter_content(chunk_size=64 * 1024):
                content.extend(chunk)

                if len(content) > _MAX_PDF_BYTES:
                    raise ValueError("PDF exceeds 50 MB size limit")

            return bytes(content)

        except Exception as exc:
            last_error = exc

            logger.warning(
                "PDF download attempt %d failed: %s",
                attempt + 1,
                exc,
            )

    raise last_error


def fetch_and_parse(state: AgentState) -> dict:
    """Download the selected paper and extract its text."""

    paper = state["selected_paper"]

    try:
        pdf_bytes = _download_pdf(paper.pdf_url)

    except Exception as exc:
        return {
            "parse_ok": False,
            "error": f"Failed to download PDF: {exc}",
        }

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:

            pdf_path = Path(tmp_dir) / "paper.pdf"
            pdf_path.write_bytes(pdf_bytes)

            doc = fitz.open(pdf_path)

    except Exception as exc:
        return {
            "parse_ok": False,
            "error": f"Failed to open PDF: {exc}",
        }

    try:
        page_count = doc.page_count

        truncated = page_count > _MAX_PAGES
        pages_to_read = min(page_count, _MAX_PAGES)

        text_parts = []

        for i in range(pages_to_read):
            page = doc.load_page(i)
            text_parts.append(page.get_text())

        raw_text = "\n".join(text_parts)

    finally:
        doc.close()

    if len(raw_text.strip()) < _MIN_TEXT_CHARS:
        return {
            "parse_ok": False,
            "raw_text": raw_text,
            "error": (
                "Very little text could be extracted. "
                "The PDF may be scanned or image-only."
            ),
        }

    return {
        "parse_ok": True,
        "raw_text": raw_text,
        "truncated": truncated,
    }


def route_parse(state: AgentState) -> str:
    """Route based on whether PDF parsing succeeded."""

    if state["parse_ok"]:
        return "parse_ok"

    return "handle_parse_failure"