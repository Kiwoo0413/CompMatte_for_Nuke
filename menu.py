"""
CompMatte for Nuke Menu Registration (menu.py)
Integrates CompMatte Node into Nuke's Toolbar, Keyer Menu, and Main Menu Bar.
"""

import os
import sys
import nuke

_curr_dir = os.path.dirname(os.path.abspath(__file__))
if _curr_dir not in sys.path:
    sys.path.insert(0, _curr_dir)

import compmatte_bridge

# 1. Add to Nodes Toolbar
toolbar = nuke.menu('Nodes')
cm_menu = toolbar.addMenu('CompMatte', icon='CompMatte.png')

# Primary command: Create CompMatte Node (Ctrl+Alt+M)
cm_menu.addCommand(
    'Create CompMatte Node',
    'import compmatte_bridge; compmatte_bridge.create_compmatte_node()',
    icon='CompMatte.png',
    shortcut='Ctrl+Alt+M'
)

# 2. Add to Keyer Menu
keyer_menu = toolbar.findItem('Keyer')
if keyer_menu:
    keyer_menu.addCommand(
        'CompMatte',
        'import compmatte_bridge; compmatte_bridge.create_compmatte_node()',
        icon='CompMatte.png'
    )

# 3. Add to Main Menu Bar
main_menubar = nuke.menu('Nuke')
top_menu = main_menubar.addMenu('&CompMatte')
top_menu.addCommand(
    'Create CompMatte Node',
    'import compmatte_bridge; compmatte_bridge.create_compmatte_node()',
    shortcut='Ctrl+Alt+M'
)
top_menu.addSeparator()
top_menu.addCommand(
    'About CompMatte for Nuke',
    """import nuke; nuke.message("CompMatte for Nuke (v3.0)\\n\\nHollywood VFX Optical Alpha Matting Toolkit.\\n\\n• Nuke IBK Clean Plate & Color Difference Keyer\\n• Topological Hole-Filling (Pure White 1.0 Core Lock)\\n• Safe Zone Edge Detail Re-Injection (100% Micro Hair Preservation)\\n• Zero PyTorch / Zero VRAM Overhead (Pure Optical 60fps+)")"""
)
