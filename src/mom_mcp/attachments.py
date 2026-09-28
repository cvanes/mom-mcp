"""Loading of local file attachments into chat message content parts.

Attachments are passed as local file paths because the server always runs
on the same machine as the calling agent. UTF-8-decodable files are inlined
as text with a filename header; images are embedded as base64 data URIs;
other documents (PDF, Word, PowerPoint, Excel, ...) are converted to
markdown with markitdown.
"""

import base64
import mimetypes
from dataclasses import dataclass
from pathlib import Path

from markitdown import MarkItDown


@dataclass(frozen=True)
class Attachment:
    path: Path
    mime_type: str

    @property
    def is_image(self) -> bool:
        return self.mime_type.startswith("image/")


def load_attachment(path_str: str) -> Attachment:
    path = Path(path_str).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Attachment not found: {path_str}")
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return Attachment(path=path, mime_type=mime_type)


def to_content_part(attachment: Attachment) -> dict:
    """Convert an attachment into an OpenAI-style chat content part."""
    if attachment.is_image:
        encoded = base64.b64encode(attachment.path.read_bytes()).decode()
        data_uri = f"data:{attachment.mime_type};base64,{encoded}"
        return {"type": "image_url", "image_url": {"url": data_uri}}
    return {"type": "text", "text": _format_text_attachment(attachment)}


def build_user_message(prompt: str, attachment_paths: list[str]) -> dict:
    """Build a single user message combining the prompt and any attachments."""
    if not attachment_paths:
        return {"role": "user", "content": prompt}
    parts = [{"type": "text", "text": prompt}]
    parts += [to_content_part(load_attachment(p)) for p in attachment_paths]
    return {"role": "user", "content": parts}


def _format_text_attachment(attachment: Attachment) -> str:
    content = _read_text(attachment.path)
    if content is None:
        content = _convert_to_markdown(attachment.path)
    return f"<attachment name=\"{attachment.path.name}\">\n{content}\n</attachment>"


def _read_text(path: Path) -> str | None:
    """Read a file as UTF-8 text, returning None for binary content."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None


def _convert_to_markdown(path: Path) -> str:
    try:
        return MarkItDown().convert(str(path)).text_content
    except Exception as exc:
        raise ValueError(
            f"Unsupported attachment type {path.name!r}: not UTF-8 text "
            f"and markitdown could not convert it"
        ) from exc
