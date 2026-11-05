"""Smoke test: the Streamlit app starts without raising an exception."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py"


def test_app_loads_without_exceptions():
    at = AppTest.from_file(str(APP_PATH), default_timeout=90).run()
    assert not at.exception