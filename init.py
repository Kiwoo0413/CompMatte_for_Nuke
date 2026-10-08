"""
CompMatte for Nuke Package Initialization (init.py)
Ensures CompMatte modules are readily importable across all Nuke sessions.
"""

import os
import sys

_curr_dir = os.path.dirname(os.path.abspath(__file__))
if _curr_dir not in sys.path:
    sys.path.insert(0, _curr_dir)
