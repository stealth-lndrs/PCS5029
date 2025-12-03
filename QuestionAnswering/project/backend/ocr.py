import logging
from typing import List, Optional

import numpy as np

try:
    import pdfplumber
except ImportError:  # pragma: no cover - optional dependency
    pdfplumber = None  # type: ignore

try:
    from pdf2image import convert_from_path
except ImportError:  # pragma: no cover - optional dependency
    convert_from_path = None  # type: ignore

try:
    from paddleocr import PaddleOCR
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "PaddleOCR is required for OCR capabilities. "
        "Install with `pip install paddleocr`."
    ) from exc


LOGGER = logging.getLogger(__name__)

OCR_LANG = "multilingual"
OCR_DPI = 300
MIN_CHAR_THRESHOLD = 40

_OCR_INSTANCE: Optional[PaddleOCR] = None


def _get_ocr() -> PaddleOCR:
    """Instantiate PaddleOCR lazily to avoid heavy startup cost."""
    global _OCR_INSTANCE
    if _OCR_INSTANCE is None:
        LOGGER.debug("Initializing PaddleOCR with lang=%s", OCR_LANG)
        _OCR_INSTANCE = PaddleOCR(
            use_angle_cls=True,
            lang=OCR_LANG,
            show_log=False,
        )
    return _OCR_INSTANCE


def _clean_text(text: str) -> str:
    """Ensure text is UTF-8 clean and stripped of noisy whitespace."""
    if not text:
        return ""
    cleaned = text.replace("\x00", " ").strip()
    return cleaned.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")


def is_page_scanned(text: str, min_chars: int = MIN_CHAR_THRESHOLD) -> bool:
    """
    Heuristically determine if a PDF page is likely scanned.

    A page is considered scanned when the available text layer is too short.
    """
    normalized = (text or "").strip()
    is_scanned = len(normalized) < max(min_chars, 1)
    LOGGER.debug(
        "Page scanned heuristic: len=%s threshold=%s -> %s",
        len(normalized),
        min_chars,
        is_scanned,
    )
    return is_scanned


def _extract_text_layer(pdf_path: str, page_number: int) -> str:
    if pdfplumber is None:
        LOGGER.warning("pdfplumber not installed; skipping text-layer extraction.")
        return ""

    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_number < 0 or page_number >= len(pdf.pages):
                raise IndexError(
                    f"Page number {page_number} out of range (0-{len(pdf.pages) - 1})"
                )
            raw_text = pdf.pages[page_number].extract_text() or ""
            return _clean_text(raw_text)
    except FileNotFoundError:
        LOGGER.error("PDF file not found: %s", pdf_path)
        raise
    except Exception as exc:
        LOGGER.exception(
            "Failed to extract text layer from %s page %s: %s",
            pdf_path,
            page_number,
            exc,
        )
        return ""


def _run_ocr(
    pdf_path: str,
    page_number: int,
    dpi: int = OCR_DPI,
) -> str:
    if convert_from_path is None:
        LOGGER.error(
            "pdf2image is required for OCR fallback. Install with `pip install pdf2image`."
        )
        return ""

    try:
        images = convert_from_path(
            pdf_path,
            dpi=dpi,
            first_page=page_number + 1,
            last_page=page_number + 1,
            thread_count=1,
            fmt="png",
            single_file=True,
        )
    except Exception as exc:
        LOGGER.exception(
            "Failed to render PDF %s page %s for OCR: %s",
            pdf_path,
            page_number,
            exc,
        )
        return ""

    if not images:
        LOGGER.warning("No rasterized image produced for %s page %s.", pdf_path, page_number)
        return ""

    image = np.array(images[0].convert("RGB"))

    try:
        ocr_engine = _get_ocr()
        result = ocr_engine.ocr(image, cls=True)
    except Exception as exc:
        LOGGER.exception("PaddleOCR failed on %s page %s: %s", pdf_path, page_number, exc)
        return ""

    lines: List[str] = []
    for block in result or []:
        for entry in block:
            try:
                lines.append(entry[1][0])
            except (IndexError, TypeError):
                continue

    joined = "\n".join(lines)
    if not joined:
        LOGGER.info("OCR returned no text for %s page %s.", pdf_path, page_number)
    return _clean_text(joined)


def extract_page_text(pdf_path: str, page_number: int) -> str:
    """
    Extract text for a specific page, falling back to OCR when necessary.
    """
    LOGGER.debug("Extracting text from %s page %s", pdf_path, page_number)
    text_layer = _extract_text_layer(pdf_path, page_number)

    if is_page_scanned(text_layer):
        LOGGER.info("Page %s appears scanned; running OCR.", page_number)
        ocr_text = _run_ocr(pdf_path, page_number)
        return ocr_text or text_layer

    return text_layer
