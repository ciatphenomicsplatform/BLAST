"""
BLAST MLOps — Shared utilities
==============================
Single source of truth for:
  - BLAST_ROOT / MLFLOW_URI  (project-wide constants)
  - inject_css()             (dark-theme stylesheet for every page)
  - StreamCapture            (redirect stdout → queue for live log streaming)

Import in any page or component:
    from components.shared import inject_css, StreamCapture, MLFLOW_URI, BLAST_ROOT
"""

import os
import sys
import queue

# ── Project root (app/components/ → app/ → BLAST/) ───────────────────────────
BLAST_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

# ── Single MLflow URI ─────────────────────────────────────────────────────────
MLFLOW_URI = "sqlite:///" + os.path.join(BLAST_ROOT, "mlruns.db")


# ── CSS ───────────────────────────────────────────────────────────────────────
_BASE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Layout ── */
.stApp { background: #0d1117; color: #e6edf3; }
section[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0d1117 0%, #161b22 100%);
}
section[data-testid="stSidebar"] * { color: #e6edf3 !important; }

/* ── Panels / cards ── */
.panel {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
}
.blast-card {
    background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 16px;
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.blast-card:hover { transform: translateY(-2px); border-color: #58a6ff; }

/* ── Hero ── */
.hero-banner {
    background: linear-gradient(135deg, #0d1117 0%, #1a2332 50%, #0d1117 100%);
    border: 1px solid #1f6feb;
    border-radius: 16px;
    padding: 48px 32px;
    text-align: center;
    margin-bottom: 32px;
}
.hero-title {
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(90deg, #58a6ff, #3fb950, #f78166);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.hero-sub { font-size: 1.1rem; color: #8b949e; margin-top: 8px; }

/* ── Badges ── */
.badge-green  { background:#1a3a1a; color:#3fb950; border:1px solid #2ea043;
                border-radius:20px; padding:2px 12px; font-size:.8rem; }
.badge-blue   { background:#0d2540; color:#58a6ff; border:1px solid #1f6feb;
                border-radius:20px; padding:2px 12px; font-size:.8rem; }
.badge-orange { background:#2d1a06; color:#e3b341; border:1px solid #9e6a03;
                border-radius:20px; padding:2px 12px; font-size:.8rem; }

/* ── Metric pill ── */
.metric-pill {
    display: inline-block;
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.85rem;
    color: #58a6ff;
    margin: 2px;
}

/* ── Log box ── */
.log-box {
    background: #010409;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 16px;
    font-family: 'Courier New', monospace;
    font-size: 0.82rem;
    color: #58a6ff;
    max-height: 340px;
    overflow-y: auto;
    white-space: pre-wrap;
}

/* ── Run rows (MLflow page) ── */
.run-row {
    background: #1c2128;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 8px;
}
.run-row:hover { border-color: #58a6ff; }

/* ── Default buttons (blue) — pages override this with a small extra block ── */
.stButton > button {
    background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* ── Metrics default colour (blue; pages can override) ── */
div[data-testid="stMetricValue"] { color: #58a6ff !important; }
h1, h2, h3 { color: #e6edf3 !important; }
"""


def inject_css(extra_css: str = "") -> None:
    """
    Inject the shared dark-theme stylesheet into the current Streamlit page.

    Args:
        extra_css: Optional additional CSS string appended after the base block.
                   Use this for per-page button-colour or metric-colour overrides.

    Example (ML page, green buttons)::

        inject_css(\"\"\"
            .stButton > button {
                background: linear-gradient(135deg, #145a2e, #2ea043) !important;
            }
            div[data-testid=\"stMetricValue\"] { color: #3fb950 !important; }
        \"\"\")
    """
    import streamlit as st

    combined = _BASE_CSS + "\n" + extra_css if extra_css else _BASE_CSS
    st.markdown(f"<style>{combined}</style>", unsafe_allow_html=True)


# ── StreamCapture ─────────────────────────────────────────────────────────────
class StreamCapture:
    """
    Context-manager that redirects stdout into a queue for live log streaming.

    Usage::

        log_q = queue.Queue()
        with StreamCapture(log_q):
            print("This appears in the Streamlit log box")
    """

    def __init__(self, log_queue: queue.Queue):
        self.queue = log_queue
        self._orig_stdout = None

    def write(self, text: str) -> None:
        if text.strip():
            self.queue.put(text)
        if self._orig_stdout:
            self._orig_stdout.write(text)

    def flush(self) -> None:
        if self._orig_stdout:
            self._orig_stdout.flush()

    def __enter__(self) -> "StreamCapture":
        self._orig_stdout = sys.stdout
        sys.stdout = self
        return self

    def __exit__(self, *args) -> None:
        sys.stdout = self._orig_stdout
