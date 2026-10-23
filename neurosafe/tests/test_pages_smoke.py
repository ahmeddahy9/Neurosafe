"""Run every Streamlit page headlessly (streamlit.testing) with and without data."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGES = ["app.py"] + sorted(str(p.relative_to(ROOT)) for p in (ROOT / "pages").glob("*.py"))

RUNNER = """
import sys
from streamlit.testing.v1 import AppTest
at = AppTest.from_file(sys.argv[1], default_timeout=90).run()
errs = [str(e.value) for e in at.exception]
print("EXCEPTIONS:", errs)
sys.exit(1 if errs else 0)
"""


def _run(page, env_extra):
    env = {**os.environ, **env_extra, "PYTHONPATH": str(ROOT)}
    return subprocess.run([sys.executable, "-c", RUNNER, page], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=240)


@pytest.mark.parametrize("page", PAGES)
def test_page_runs_without_data(page, tmp_path):
    r = _run(page, {"NEUROSAFE_DATA_DIR": str(tmp_path / "d"), "NEUROSAFE_MODELS_DIR": str(tmp_path / "m")})
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.skipif(not (ROOT / "data/processed/reports.csv").exists(), reason="run the pipeline first")
@pytest.mark.parametrize("page", PAGES)
def test_page_runs_with_data(page):
    r = _run(page, {})
    assert r.returncode == 0, r.stdout + r.stderr
