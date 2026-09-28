"""Tests for attachment loading: text, images, markitdown conversion and errors."""

import base64
from unittest.mock import patch

import pytest

from mom_mcp.attachments import build_user_message, load_attachment, to_content_part


def test_text_file_is_inlined_with_a_filename_header(tmp_path):
    # Arrange
    note = tmp_path / "notes.txt"
    note.write_text("hello world", encoding="utf-8")

    # Act
    part = to_content_part(load_attachment(str(note)))

    # Assert
    assert part == {
        "type": "text",
        "text": '<attachment name="notes.txt">\nhello world\n</attachment>',
    }


def test_image_is_embedded_as_a_base64_data_uri(tmp_path):
    # Arrange
    image = tmp_path / "diagram.png"
    image.write_bytes(b"\x89PNG fake")

    # Act
    part = to_content_part(load_attachment(str(image)))

    # Assert
    encoded = base64.b64encode(b"\x89PNG fake").decode()
    assert part == {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{encoded}"},
    }


def test_binary_document_is_converted_with_markitdown(tmp_path):
    # Arrange
    doc = tmp_path / "report.pdf"
    doc.write_bytes(b"%PDF-1.7 binary \x00\xff")

    # Act
    with patch("mom_mcp.attachments.MarkItDown") as markitdown:
        markitdown.return_value.convert.return_value.text_content = "# Report"
        part = to_content_part(load_attachment(str(doc)))

    # Assert
    assert part == {
        "type": "text",
        "text": '<attachment name="report.pdf">\n# Report\n</attachment>',
    }
    markitdown.return_value.convert.assert_called_once_with(str(doc))


def test_unconvertible_binary_raises_a_clear_error(tmp_path):
    # Arrange
    blob = tmp_path / "data.bin"
    blob.write_bytes(b"\x00\x01\x02\xff")

    # Act / Assert
    with patch("mom_mcp.attachments.MarkItDown") as markitdown:
        markitdown.return_value.convert.side_effect = Exception("no converter")
        with pytest.raises(ValueError, match="Unsupported attachment type"):
            to_content_part(load_attachment(str(blob)))


def test_missing_file_raises_file_not_found(tmp_path):
    # Act / Assert
    with pytest.raises(FileNotFoundError, match="Attachment not found"):
        load_attachment(str(tmp_path / "nope.txt"))


def test_build_user_message_without_attachments_returns_a_plain_string():
    # Act
    message = build_user_message("just a prompt", [])

    # Assert
    assert message == {"role": "user", "content": "just a prompt"}


def test_build_user_message_combines_prompt_and_attachments(tmp_path):
    # Arrange
    note = tmp_path / "notes.txt"
    note.write_text("context", encoding="utf-8")

    # Act
    message = build_user_message("review this", [str(note)])

    # Assert
    assert message["role"] == "user"
    assert [part["type"] for part in message["content"]] == ["text", "text"]
    assert message["content"][0] == {"type": "text", "text": "review this"}
