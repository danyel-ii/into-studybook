import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
API_DIR = os.path.join(BASE_DIR, "apps", "api")
if API_DIR not in sys.path:
    sys.path.append(API_DIR)

from app.main import app  # noqa: E402
