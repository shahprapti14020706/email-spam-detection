"""Meaningful frontend smoke checks using Streamlit's built-in AppTest."""
import os
from pathlib import Path
from unittest.mock import patch
from streamlit.testing.v1 import AppTest

BASE = Path(__file__).resolve().parent

def verify():
    app = AppTest.from_file(str(BASE / "app.py"), default_timeout=60).run()
    assert not app.exception, app.exception
    assert [tab.label for tab in app.tabs] == ["Check Message", "Model Performance", "How It Works"]
    app.button[2].click().run()
    assert "Please enter" in app.warning[0].value
    app.button[0].click().run()
    app.button[2].click().run()
    assert any(e.value.startswith("Spam —") for e in app.error), "Spam example prediction"
    assert any("Estimated Spam" in metric.label for metric in app.metric)
    app.button[1].click().run()
    app.button[2].click().run()
    assert any(e.value.startswith("Not Spam —") for e in app.success), "Normal example prediction"
    app.text_area[0].set_value("the and !!!").run()
    app.button[2].click().run()
    assert any("No recognised features" in e.value for e in app.warning)
    assert not app.success and not app.error
    app.text_area[0].set_value("zzzxxyyqqq").run()
    app.button[2].click().run()
    assert any("No recognised features" in e.value for e in app.warning)
    assert not app.exception
    previous = Path.cwd()
    try:
        os.chdir(BASE.parent)
        other = AppTest.from_file(str(BASE / "app.py"), default_timeout=60).run()
        assert not other.exception and len(other.tabs) == 3
    finally:
        os.chdir(previous)
    real_stat = Path.stat
    def absent_artifact(path, *args, **kwargs):
        if path == BASE / "artifacts" / "spam_model.pkl":
            raise FileNotFoundError("Simulated absent model")
        return real_stat(path, *args, **kwargs)
    with patch.object(Path, "stat", absent_artifact):
        missing = AppTest.from_file(str(BASE / "app.py"), default_timeout=60).run()
        assert not missing.exception
        assert any("python train.py" in e.value for e in missing.error)
    print("PASS: tabs, spam/ham examples, estimates, empty input, no features, file-relative paths, missing artifacts")

if __name__ == "__main__":
    verify()
