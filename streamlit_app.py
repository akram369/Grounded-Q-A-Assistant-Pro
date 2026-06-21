import sys
from pathlib import Path

# Add the 'src' directory to the Python path to ensure document_qa is importable
src_dir = Path(__file__).parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Import and run the UI script
import document_qa.ui
