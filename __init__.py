"""
CompMatte for Nuke Package.
Pure Optical VFX Matting Toolkit for Foundry Nuke.
"""

from .compmatte_core import (
    CompMatteConfig,
    IBKEngine,
    CoreEngine,
    MatteFusionEngine,
    ScreenType,
)
from .compmatte_bridge import create_compmatte_node

__version__ = "3.0.0"
__all__ = [
    "CompMatteConfig",
    "IBKEngine",
    "CoreEngine",
    "MatteFusionEngine",
    "ScreenType",
    "create_compmatte_node",
]
