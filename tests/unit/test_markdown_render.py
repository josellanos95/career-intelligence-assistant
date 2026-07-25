from app.domain.markdown_render import render_markdown


def test_renders_bold_and_lists():
    html = render_markdown("**Matched skills**\n\n- Python\n- FastAPI")
    assert "<strong>Matched skills</strong>" in html
    assert "<ul>" in html
    assert "<li>Python</li>" in html


def test_renders_numbered_lists_and_headers():
    html = render_markdown("### Overall fit\n\n1. First point\n2. Second point")
    assert "<h3>Overall fit</h3>" in html
    assert "<ol>" in html


def test_strips_disallowed_tags():
    html = render_markdown("Hello <script>alert('xss')</script> world")
    # The tag itself is stripped (no executable script survives); the inert
    # text content it wrapped is left behind as plain text, which is safe.
    assert "<script>" not in html
    assert "</script>" not in html
