import sys
import os

# Add src/ to PYTHONPATH for pytest collection
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
