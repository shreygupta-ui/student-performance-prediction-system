"""Headless smoke test of every page. Run: python tests/test_app.py"""
from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")

PAGES = ["🏠 Home", "🔮 Student Prediction", "📊 Data Analysis",
         "🧪 Model Evaluation", "ℹ️ About Project"]
for p in PAGES:
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value(p).run()
    assert not at.exception, f"{p}: {at.exception}"
    assert not at.error, f"{p}: error shown: {[e.value for e in at.error]}"
    print("OK page:", p)

at = AppTest.from_file(APP, default_timeout=60).run()
at.sidebar.radio[0].set_value("🔮 Student Prediction").run()
at.button[0].click().run()
assert not at.exception, at.exception
print("OK prediction click ->", [m.value[:60] for m in at.markdown if "Predicted" in m.value][:1])
