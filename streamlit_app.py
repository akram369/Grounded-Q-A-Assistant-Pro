import sys
from pathlib import Path

# Override SQLite with pysqlite3-binary for ChromaDB compatibility on Streamlit Cloud
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except Exception:
    pass

# Add the 'src' directory to the Python path to ensure document_qa is importable
src_dir = Path(__file__).parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Force reload of document_qa.ui to ensure it re-runs on every user interaction
if "document_qa.ui" in sys.modules:
    del sys.modules["document_qa.ui"]
import document_qa.ui

