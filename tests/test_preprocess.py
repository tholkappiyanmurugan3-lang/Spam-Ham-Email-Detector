"""
tests/test_preprocess.py
========================
Unit tests for the email preprocessing pipeline.
"""

import pytest
from spam_detector.preprocess import (
    clean_email,
    clean_batch,
    strip_html,
    remove_urls,
    remove_headers,
    collapse_whitespace,
)


class TestStripHtml:
    def test_removes_basic_tags(self):
        result = strip_html("<b>Hello</b> <i>World</i>")
        assert "<b>" not in result
        assert "Hello" in result
        assert "World" in result

    def test_handles_full_html_document(self):
        html = "<html><head><title>Spam</title></head><body><p>Buy now!</p></body></html>"
        result = strip_html(html)
        assert "Buy now!" in result
        assert "<" not in result

    def test_handles_plain_text(self):
        result = strip_html("No HTML here.")
        assert result.strip() == "No HTML here."


class TestRemoveUrls:
    def test_removes_http(self):
        result = remove_urls("Visit http://spam.com for offers")
        assert "http://spam.com" not in result
        assert "<URL>" in result

    def test_removes_https(self):
        result = remove_urls("See https://example.com/page?q=1")
        assert "https://example.com" not in result

    def test_preserves_surrounding_text(self):
        result = remove_urls("Click http://x.com now!")
        assert "Click" in result
        assert "now!" in result


class TestRemoveHeaders:
    def test_removes_from_header(self):
        text = "From: sender@example.com\nHello there!"
        result = remove_headers(text)
        assert "From:" not in result
        assert "Hello there!" in result

    def test_removes_subject_header(self):
        text = "Subject: You won a prize\nBody text here."
        result = remove_headers(text)
        assert "Subject:" not in result


class TestCollapseWhitespace:
    def test_collapses_multiple_spaces(self):
        result = collapse_whitespace("hello   world")
        assert result == "hello world"

    def test_collapses_newlines(self):
        result = collapse_whitespace("line1\n\n\nline2")
        assert result == "line1 line2"

    def test_strips_edges(self):
        result = collapse_whitespace("  padded  ")
        assert result == "padded"


class TestCleanEmail:
    def test_full_pipeline(self):
        raw = (
            "From: spam@spam.com\n"
            "<html><body>WIN FREE CASH at http://win.com!</body></html>"
        )
        result = clean_email(raw)
        assert "From:" not in result
        assert "<html>" not in result
        assert "http://win.com" not in result
        assert "win free cash" in result  # lowercased

    def test_empty_string_returns_empty(self):
        assert clean_email("") == ""

    def test_non_string_returns_empty(self):
        assert clean_email(None) == ""  # type: ignore

    def test_lowercase(self):
        result = clean_email("HELLO WORLD", lowercase=True)
        assert result == "hello world"

    def test_no_lowercase(self):
        result = clean_email("HELLO", lowercase=False)
        assert "HELLO" in result


class TestCleanBatch:
    def test_processes_list(self):
        texts = ["Hello <b>World</b>", "Visit http://example.com"]
        results = clean_batch(texts)
        assert len(results) == 2
        assert "<b>" not in results[0]
        assert "http://example.com" not in results[1]

    def test_empty_list(self):
        assert clean_batch([]) == []
