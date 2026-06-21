import sys
from pathlib import Path

# Override SQLite with pysqlite3-binary for ChromaDB compatibility on Streamlit Cloud
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

# Add the 'src' directory to the Python path to ensure document_qa is importable
src_dir = Path(__file__).parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Import and run the UI script
import document_qa.ui
