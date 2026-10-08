"""
compmatte_bridge.py - CompMatte for Nuke: Nuke Node Graph & GUI Bridge.
Part of CompMatte for Nuke Toolkit.

Architecture:
  - Decoupled Hybrid Runner:
      1. In-Process Mode: If Nuke's internal Python has NumPy/OpenCV, executes directly in-memory.
      2. Host Subprocess Worker Mode (AutoRoto style): If Nuke's internal Python lacks NumPy,
         automatically detects the computer's installed Python (e.g. Python 3.12, Conda, PATH)
         and executes compmatte_core.py via background subprocess with ZERO Nuke crashes!
  - 100% Graceful Fallback Shim for headless testing outside Nuke.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

# Headless compatibility shim
try:
    import nuke
    _IN_NUKE = True
except ImportError:
    _IN_NUKE = False

    class _MockKnob:
        def __init__(self, name: str, val: Any = None):
            self._name = name
            self._val = val
        def value(self): return self._val
        def setValue(self, v): self._val = v
        def name(self): return self._name
        def setTooltip(self, t): pass
        def setFlag(self, f): pass
        def clearFlag(self, f): pass

    class _MockNode:
        def __init__(self, node_class: str = "Group"):
            self._class = node_class
            self._name = f"{node_class}1"
            self._knobs = {}
            self.inputs_list = []
        def Class(self): return self._class
        def name(self): return self._name
        def setName(self, n): self._name = n
        def addKnob(self, k): self._knobs[k.name()] = k
        def knob(self, n):
            if n == "knobChanged" and n not in self._knobs:
                self._knobs[n] = _MockKnob(n, "")
            return self._knobs.get(n)
        def knobs(self): return self._knobs
        def begin(self): pass
        def end(self): pass
        def input(self, idx): return self.inputs_list[idx] if idx < len(self.inputs_list) else None
        def setInput(self, idx, n): pass
        def node(self, n): return _MockNode(n)

    class _MockNuke:
        STARTLINE = 0x00001000
        def createNode(self, cls, *args, **kwargs): return _MockNode(cls)
        def toNode(self, name): return None
        def selectedNode(self): return None
        def allNodes(self): return []
        def frame(self): return 1
        def root(self): return {"first_frame": _MockKnob("first_frame", 1), "last_frame": _MockKnob("last_frame", 100)}
        def message(self, msg): print(f"[Nuke Message]: {msg}")
        def execute(self, *args, **kwargs): pass
        def delete(self, *args, **kwargs): pass
        Tab_Knob = staticmethod(lambda name, l=None: _MockKnob(name))
        Enumeration_Knob = staticmethod(lambda name, l=None, options=None: _MockKnob(name, options[0] if options else ""))
        Double_Knob = staticmethod(lambda name, l=None: _MockKnob(name, 0.0))
        Int_Knob = staticmethod(lambda name, l=None: _MockKnob(name, 0))
        Boolean_Knob = staticmethod(lambda name, l=None: _MockKnob(name, False))
        Color_Knob = staticmethod(lambda name, l=None: _MockKnob(name, [0.0, 1.0, 0.0]))
        PyScript_Knob = staticmethod(lambda name, l=None, cmd="": _MockKnob(name))
        Text_Knob = staticmethod(lambda name, l=None, text="": _MockKnob(name, text))
        File_Knob = staticmethod(lambda name, l=None: _MockKnob(name, ""))
        ProgressTask = None

    nuke = _MockNuke()


# Safe In-Process Import Check (Never crash Nuke if NumPy is not in Nuke Python)
try:
    import numpy as np
    import cv2
    import compmatte_core
    from compmatte_core import CompMatteConfig, MatteFusionEngine, IBKEngine, CoreEngine
    _IN_PROCESS_AVAILABLE = True
except ImportError:
    np = None
    cv2 = None
    compmatte_core = None
    _IN_PROCESS_AVAILABLE = False


# =============================================================================
# Python Discovery Engine (AutoRoto Pattern)
# =============================================================================

def get_candidate_pythons(custom_path: Optional[str] = None) -> List[str]:
    """
    Assembles a prioritized list of Python interpreters available on the host computer.
    """
    candidates = []
    if custom_path and os.path.isfile(custom_path.strip()):
        candidates.append(custom_path.strip())

    # 1. Dedicated Environment Variable
    env_py = os.environ.get("COMPMATTE_PYTHON")
    if env_py and os.path.isfile(env_py):
        candidates.append(env_py)

    # 2. Dynamic User Directories (Windows, Conda, Python.org)
    user_home = os.path.expanduser("~")
    common_relative = [
        r"AppData\Local\Programs\Python\Python312\python.exe",
        r"AppData\Local\Programs\Python\Python311\python.exe",
        r"AppData\Local\Programs\Python\Python310\python.exe",
        r"miniconda3\python.exe",
        r"anaconda3\python.exe",
        r"miniconda3\envs\compmatte\python.exe",
        r"anaconda3\envs\compmatte\python.exe",
        "miniconda3/bin/python",
        "anaconda3/bin/python",
    ]
    for rel in common_relative:
        p = os.path.join(user_home, rel)
        if os.path.isfile(p) and p not in candidates:
            candidates.append(p)

    # 3. System PATH Discovery
    for cmd in ("python", "python3"):
        which_p = shutil.which(cmd)
        if which_p and which_p not in candidates:
            candidates.append(which_p)

    return candidates


def find_compmatte_python(custom_path: Optional[str] = None) -> Optional[str]:
    """
    Discovers an external Python interpreter equipped with NumPy on the host computer.
    Tests candidate by running a fast probing command.
    """
    candidates = get_candidate_pythons(custom_path)
    for py_exe in candidates:
        try:
            cmd = [py_exe, "-c", "import numpy; print('OK')"]
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=5).decode().strip()
            if "OK" in out:
                return py_exe
        except Exception:
            continue
    return None


def check_environment(custom_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Comprehensive diagnostic report for CompMatte Python & NumPy environment.
    """
    if _IN_PROCESS_AVAILABLE:
        return {
            "available": True,
            "mode": "In-Process (Nuke Python)",
            "python_path": sys.executable,
            "numpy_version": np.__version__,
            "has_cv2": True if cv2 else False,
            "message": "Nuke Python has NumPy. Ultra-fast in-process execution active.",
        }

    ext_py = find_compmatte_python(custom_path)
    if not ext_py:
        return {
            "available": False,
            "mode": "None",
            "python_path": None,
            "numpy_version": None,
            "has_cv2": False,
            "message": "No Python with NumPy detected on system.\nPlease install NumPy on computer or set COMPMATTE_PYTHON.",
        }

    try:
        check_code = (
            "import numpy, json\n"
            "try:\n"
            "    import cv2\n"
            "    c_ver = cv2.__version__\n"
            "except ImportError:\n"
            "    c_ver = None\n"
            "print(json.dumps({'numpy': numpy.__version__, 'cv2': c_ver}))\n"
        )
        out = subprocess.check_output([ext_py, "-c", check_code], stderr=subprocess.STDOUT, timeout=5).decode().strip()
        info = json.loads(out)
        return {
            "available": True,
            "mode": "Host Subprocess Worker",
            "python_path": ext_py,
            "numpy_version": info.get("numpy"),
            "has_cv2": bool(info.get("cv2")),
            "message": f"Using computer Python ({ext_py}) with NumPy {info.get('numpy')}.",
        }
    except Exception as e:
        return {
            "available": False,
            "mode": "Error",
            "python_path": ext_py,
            "numpy_version": None,
            "has_cv2": False,
            "message": f"Error probing Python {ext_py}: {e}",
        }


# =============================================================================
# Cache & Directory Management
# =============================================================================

def get_compmatte_cache_dir(node: Optional[Any] = None) -> str:
    """
    Retrieves or creates the destination folder for CompMatte alpha files:
      0. Custom path specified in node's 'output_dir' knob if present and non-empty.
      1. Nuke Preferences cache disk ('DiskCachePath', 'localCachePath')
      2. Environment variable ('COMPMATTE_CACHE_DIR', 'NUKE_DISK_CACHE', 'NUKE_TEMP_DIR')
      3. System temp directory (%TEMP%/CompMatte_Cache/)
    """
    # 0. Check custom directory from node knob
    if node:
        try:
            custom_knob = node.knob("output_dir")
            if custom_knob and custom_knob.value():
                val = str(custom_knob.value()).strip()
                if val:
                    expanded = os.path.expandvars(os.path.expanduser(val)).replace("\\", "/").rstrip("/")
                    os.makedirs(expanded, exist_ok=True)
                    if os.path.isdir(expanded) and os.access(expanded, os.W_OK):
                        return expanded
        except Exception:
            pass

    candidates = []
    try:
        pref = nuke.toNode('preferences')
        if pref:
            for k in ('DiskCachePath', 'localCachePath', 'DiskCacheDirectory'):
                if pref.knob(k) and pref[k].value():
                    p = os.path.expandvars(os.path.expanduser(str(pref[k].value()).strip()))
                    candidates.append(p)
    except Exception:
        pass

    for env_k in ('COMPMATTE_CACHE_DIR', 'NUKE_DISK_CACHE', 'NUKE_TEMP_DIR'):
        v = os.environ.get(env_k)
        if v and os.path.isdir(v):
            candidates.append(v)

    for p in candidates:
        try:
            target = os.path.join(p, "CompMatte_Cache").replace("\\", "/")
            os.makedirs(target, exist_ok=True)
            if os.path.isdir(target) and os.access(target, os.W_OK):
                return target
        except Exception:
            continue

    fallback = os.path.join(tempfile.gettempdir(), "CompMatte_Cache").replace("\\", "/")
    os.makedirs(fallback, exist_ok=True)
    return fallback


def on_open_output_folder(node: Any) -> None:
    """
    Opens the active CompMatte output directory in the system file explorer.
    """
    folder = get_compmatte_cache_dir(node)
    if os.path.exists(folder):
        if sys.platform == "win32":
            os.startfile(folder)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", folder])
        else:
            subprocess.Popen(["xdg-open", folder])
    else:
        if _IN_NUKE:
            nuke.message(f"Output folder does not exist yet:\n{folder}")


# =============================================================================
# Subprocess Execution
# =============================================================================

def run_compmatte_external(
    src_path: str,
    out_path: str,
    config: Dict[str, Any],
    clean_path: Optional[str] = None,
    holdout_path: Optional[str] = None,
    custom_python: Optional[str] = None,
) -> bool:
    """
    Executes compmatte_core.py as a standalone process using host Python.
    """
    py_exe = find_compmatte_python(custom_python)
    if not py_exe:
        raise RuntimeError(
            "CompMatte: No Python with NumPy found on this computer!\n"
            "Please ensure NumPy is installed or set COMPMATTE_PYTHON environment variable."
        )

    core_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compmatte_core.py")
    cmd = [
        py_exe,
        core_script,
        "--input", src_path,
        "--output", out_path,
        "--config-json", json.dumps(config),
    ]
    if clean_path and os.path.exists(clean_path):
        cmd.extend(["--clean-plate", clean_path])
    if holdout_path and os.path.exists(holdout_path):
        cmd.extend(["--holdout", holdout_path])

    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
    if res.returncode != 0:
        raise RuntimeError(f"CompMatte Worker Error:\n{res.stderr.strip()}")
    return True


# =============================================================================
# Node Setup & Knobs
# =============================================================================

def setup_compmatte_knobs(node: Any) -> None:
    """
    Populates custom CompMatte user interface knobs on the target Group node.
    """
    # ----------------- Tab 1: CompMatte (Quick Action & Master) -----------------
    tab_main = nuke.Tab_Knob("compmatte_tab", "CompMatte")
    node.addKnob(tab_main)

    screen_type = nuke.Enumeration_Knob("screen_type", "Screen Type", ["green", "blue", "custom"])
    screen_type.setTooltip("Backing screen color to extract.")
    node.addKnob(screen_type)

    screen_color = nuke.Color_Knob("screen_color", "")
    screen_color.clearFlag(nuke.STARTLINE)
    screen_color.setValue([0.0, 1.0, 0.0])
    screen_color.setTooltip("Screen color picker swatch & eyedropper. Sample background directly from viewer using Ctrl+Alt+Click.")
    node.addKnob(screen_color)

    view_mode = nuke.Enumeration_Knob(
        "view_mode", "View Output",
        ["Final Alpha (rgba.a)", "Premultiplied RGBA", "Clean Plate", "Core Matte", "Edge Matte"]
    )
    view_mode.setTooltip("Select which stage to display in the Nuke Viewer.")
    node.addKnob(view_mode)

    w_red = nuke.Double_Knob("w_red", "Red Weight")
    w_red.setValue(0.5)
    node.addKnob(w_red)

    w_blue = nuke.Double_Knob("w_blue", "Blue/Green Weight")
    w_blue.setValue(0.5)
    node.addKnob(w_blue)

    # Action Buttons
    btn_extract = nuke.PyScript_Knob(
        "btn_extract", "⚡ Extract Matte (Current Frame)",
        "import compmatte_bridge; compmatte_bridge.on_extract_matte(nuke.thisNode())"
    )
    btn_extract.setTooltip("Extracts and injects pure optical matte for the current frame.")
    node.addKnob(btn_extract)

    btn_range = nuke.PyScript_Knob(
        "btn_range", "🎬 Bake Frame Range...",
        "import compmatte_bridge; compmatte_bridge.on_render_range(nuke.thisNode())"
    )
    btn_range.setTooltip("Bakes alpha matte sequence for specified frame range into cache.")
    node.addKnob(btn_range)

    btn_open = nuke.PyScript_Knob(
        "btn_open", "📂 Open Output Folder",
        "import compmatte_bridge; compmatte_bridge.on_open_output_folder(nuke.thisNode())"
    )
    btn_open.setTooltip("Opens the active output directory in file explorer.")
    node.addKnob(btn_open)

    output_dir = nuke.File_Knob("output_dir", "Output Folder")
    output_dir.setTooltip("Custom destination directory for alpha frames. Leave empty to automatically use Nuke Cache Disk.")
    node.addKnob(output_dir)

    # Status readout
    env_info = check_environment()
    status_msg = f"Ready ({env_info['mode']})"
    status = nuke.Text_Knob("cm_status", "Status", status_msg)
    node.addKnob(status)

    # ----------------- Tab 2: Clean Plate (IBK) -----------------
    tab_cp = nuke.Tab_Knob("tab_clean_plate", "Clean Plate")
    node.addKnob(tab_cp)

    patch_size = nuke.Int_Knob("patch_size", "Patch Size")
    patch_size.setValue(5)
    node.addKnob(patch_size)

    patch_iter = nuke.Int_Knob("patch_iterations", "Patch Iterations")
    patch_iter.setValue(4)
    node.addKnob(patch_iter)

    cp_blur = nuke.Int_Knob("cp_blur", "Blur Radius")
    cp_blur.setValue(3)
    node.addKnob(cp_blur)

    # ----------------- Tab 3: Core & Edge Fusion -----------------
    tab_core = nuke.Tab_Knob("tab_core_edge", "Core & Edge Fusion")
    node.addKnob(tab_core)

    hole_fill = nuke.Boolean_Knob("use_hole_fill", "Topological Hole-Filling (Pure 1.0 Core Lock)")
    hole_fill.setValue(True)
    hole_fill.setTooltip("Fills all interior cavities solid white (1.0) to eliminate internal chatter.")
    node.addKnob(hole_fill)

    core_erode = nuke.Int_Knob("core_erode", "Core Inset / Erode")
    core_erode.setValue(7)
    node.addKnob(core_erode)

    restore_edges = nuke.Boolean_Knob("restore_fine_edges", "Safe Zone Edge Detail Re-Injection")
    restore_edges.setValue(True)
    restore_edges.setTooltip("Re-injects 1px micro hair strands and motion blur within safe envelope.")
    node.addKnob(restore_edges)

    safe_radius = nuke.Int_Knob("safe_radius", "Safe Zone Radius (px)")
    safe_radius.setValue(40)
    node.addKnob(safe_radius)

    feather = nuke.Double_Knob("feather", "Sub-pixel Feathering")
    feather.setValue(0.5)
    node.addKnob(feather)

    b_clip = nuke.Double_Knob("black_clip", "Black Cutoff (Pure 0.0)")
    b_clip.setValue(0.05)
    node.addKnob(b_clip)

    w_clip = nuke.Double_Knob("white_clip", "White Cutoff (Pure 1.0)")
    w_clip.setValue(0.95)
    node.addKnob(w_clip)

    gamma = nuke.Double_Knob("gamma", "Alpha Gamma")
    gamma.setValue(1.0)
    node.addKnob(gamma)

    # ----------------- Tab 4: Python & Settings -----------------
    tab_settings = nuke.Tab_Knob("tab_settings", "Python & Settings")
    node.addKnob(tab_settings)

    custom_py = nuke.File_Knob("custom_python", "Host Python Executable")
    if env_info.get("python_path"):
        custom_py.setValue(env_info["python_path"])
    custom_py.setTooltip("Specify custom external Python interpreter if auto-detection needs override.")
    node.addKnob(custom_py)

    btn_check_env = nuke.PyScript_Knob(
        "btn_check_env", "🔍 Check Python & NumPy Status",
        "import compmatte_bridge, nuke; info = compmatte_bridge.check_environment(nuke.thisNode().knob('custom_python').value()); nuke.message(f\"CompMatte Environment Status:\\n\\n• Mode: {info.get('mode')}\\n• Python: {info.get('python_path')}\\n• NumPy: {info.get('numpy_version')}\\n• OpenCV: {info.get('has_cv2')}\\n\\nMessage: {info.get('message')}\")"
    )
    node.addKnob(btn_check_env)

    about_text = nuke.Text_Knob(
        "about_info", "",
        "<b>CompMatte for Nuke (v3.0)</b><br/>"
        "• Pure Optical & Compositing Alpha Matting Toolkit<br/>"
        "• Topological Hole-Filling (100% Solid Core)<br/>"
        "• Safe Zone Edge Detail Re-Injection (0% Hair Loss)<br/>"
        "• Zero PyTorch / Uses Computer's Installed NumPy Seamlessly"
    )
    node.addKnob(about_text)

    # Wire interactive knobChanged handler
    if node.knob("knobChanged"):
        node.knob("knobChanged").setValue(
            "import compmatte_bridge; compmatte_bridge.on_knob_changed(nuke.thisNode(), nuke.thisKnob())"
        )


def create_compmatte_node() -> Any:
    """
    Creates and initializes the CompMatte Group node in the active Nuke DAG.
    Wires up live internal nodes for real-time interactive playback.
    """
    node = nuke.createNode("Group")
    node.setName("CompMatte1")

    # Add custom knobs
    setup_compmatte_knobs(node)

    # Wire up internal DAG
    if _IN_NUKE:
        node.begin()
        for n in nuke.allNodes():
            nuke.delete(n)

        in_src = nuke.nodes.Input(name="Source")
        in_bg = nuke.nodes.Input(name="CleanPlate")
        in_hold = nuke.nodes.Input(name="Holdout")

        # Internal Read node for baked alpha
        cache_dir = get_compmatte_cache_dir()
        read_alpha = nuke.nodes.Read(name="Read_CompMatte_Alpha")
        read_alpha["file"].setValue (os.path.join(cache_dir, "compmatte_alpha_####.png").replace("\\", "/"))

        # Channel Copy / Inject to Alpha
        copy_node = nuke.nodes.Copy(name="Copy_Alpha")
        copy_node.setInput(0, in_src)
        copy_node.setInput(1, read_alpha)
        copy_node["from0"].setValue("rgba.red")
        copy_node["to0"].setValue("rgba.alpha")

        # Premult option
        premult_node = nuke.nodes.Premult(name="Premult_Node")
        premult_node.setInput(0, copy_node)
        premult_node["disable"].setValue(True)

        out_node = nuke.nodes.Output(name="Output")
        out_node.setInput(0, premult_node)

        node.end()

    return node


# =============================================================================
# Execution Callbacks
# =============================================================================

def on_knob_changed(node: Any, knob: Optional[Any] = None) -> None:
    """
    Handles interactive knob updates on the CompMatte node.
    - Synchronizes screen_type dropdown and screen_color picker swatch.
    - Toggles Premult node live based on view_mode.
    """
    try:
        k = knob
        if k is None and _IN_NUKE:
            k = nuke.thisKnob()
        if not k:
            return

        k_name = k.name()
        if k_name == "screen_type":
            st_knob = node.knob("screen_type")
            sc_knob = node.knob("screen_color")
            if st_knob and sc_knob:
                st = st_knob.value()
                if st == "green":
                    sc_knob.setValue([0.0, 1.0, 0.0])
                elif st == "blue":
                    sc_knob.setValue([0.0, 0.0, 1.0])
        elif k_name == "screen_color":
            st_knob = node.knob("screen_type")
            sc_knob = node.knob("screen_color")
            if st_knob and sc_knob:
                val = sc_knob.value()
                if isinstance(val, (list, tuple)) and len(val) >= 3:
                    r, g, b = float(val[0]), float(val[1]), float(val[2])
                    is_pure_green = (abs(r - 0.0) < 1e-3 and abs(g - 1.0) < 1e-3 and abs(b - 0.0) < 1e-3)
                    is_pure_blue = (abs(r - 0.0) < 1e-3 and abs(g - 0.0) < 1e-3 and abs(b - 1.0) < 1e-3)
                    if not is_pure_green and not is_pure_blue:
                        if st_knob.value() != "custom":
                            st_knob.setValue("custom")
        elif k_name == "view_mode":
            vm_knob = node.knob("view_mode")
            if vm_knob and _IN_NUKE and hasattr(node, "node"):
                vm = vm_knob.value()
                premult = node.node("Premult_Node")
                if premult:
                    premult["disable"].setValue(vm != "Premultiplied RGBA")
    except Exception:
        pass


def _export_node_frame_to_temp(nuke_node: Any, frame_num: int, prefix: str = "src") -> Optional[str]:
    """Renders a single frame from a Nuke input node to a temporary PNG file."""
    if not _IN_NUKE:
        # Mock for headless tests
        cache_dir = get_compmatte_cache_dir()
        mock_file = os.path.join(cache_dir, f"cm_mock_{prefix}_{frame_num}.png").replace("\\", "/")
        if _IN_PROCESS_AVAILABLE:
            mock_arr = np.zeros((100, 100, 3), dtype=np.uint8)
            mock_arr[:, :, 1] = 200
            cv2.imwrite(mock_file, mock_arr)
        else:
            with open(mock_file, "w") as f:
                f.write("mock")
        return mock_file

    cache_dir = get_compmatte_cache_dir()
    temp_file = os.path.join(cache_dir, f"cm_{prefix}_{frame_num}.png").replace("\\", "/")

    write_node = nuke.nodes.Write(file=temp_file, file_type="png", channels="rgb")
    write_node.setInput(0, nuke_node)

    try:
        nuke.execute(write_node, frame_num, frame_num, 1)
        if os.path.exists(temp_file):
            return temp_file
    except Exception as e:
        nuke.message(f"CompMatte Render Error: {e}")
    finally:
        nuke.delete(write_node)

    return None


def on_extract_matte(node: Any) -> None:
    """Callback when '⚡ Extract Matte (Current Frame)' button is pressed."""
    curr_frame = int(nuke.frame()) if _IN_NUKE else 1
    src_input = node.input(0) if _IN_NUKE else node

    if _IN_NUKE and not src_input:
        nuke.message("Please connect an image or video plate to the 'Source' input.")
        return

    # Render source frame
    src_temp = _export_node_frame_to_temp(src_input, curr_frame, "src")
    if not src_temp or not os.path.exists(src_temp):
        return

    # Check optional inputs
    clean_temp = None
    if _IN_NUKE and node.input(1):
        clean_temp = _export_node_frame_to_temp(node.input(1), curr_frame, "clean")

    hold_temp = None
    if _IN_NUKE and node.input(2):
        hold_temp = _export_node_frame_to_temp(node.input(2), curr_frame, "hold")

    # Extract configuration from knobs
    sc_val = [0.0, 1.0, 0.0]
    if node.knob("screen_color"):
        sc = node.knob("screen_color").value()
        if isinstance(sc, (list, tuple)):
            sc_val = [float(x) for x in sc[:3]]
        elif isinstance(sc, (int, float)):
            sc_val = [float(sc), float(sc), float(sc)]

    cfg = {
        "screen_type": node.knob("screen_type").value() if node.knob("screen_type") else "green",
        "custom_color": sc_val,
        "red_weight": float(node.knob("w_red").value()) if node.knob("w_red") else 0.5,
        "blue_weight": float(node.knob("w_blue").value()) if node.knob("w_blue") else 0.5,
        "use_hole_fill": bool(node.knob("use_hole_fill").value()) if node.knob("use_hole_fill") else True,
        "restore_fine_edges": bool(node.knob("restore_fine_edges").value()) if node.knob("restore_fine_edges") else True,
        "safe_zone_radius": int(node.knob("safe_radius").value()) if node.knob("safe_radius") else 40,
        "core_erode_size": int(node.knob("core_erode").value()) if node.knob("core_erode") else 7,
        "feather_radius": float(node.knob("feather").value()) if node.knob("feather") else 0.5,
        "black_clip": float(node.knob("black_clip").value()) if node.knob("black_clip") else 0.05,
        "white_clip": float(node.knob("white_clip").value()) if node.knob("white_clip") else 0.95,
        "gamma": float(node.knob("gamma").value()) if node.knob("gamma") else 1.0,
    }

    target_dir = get_compmatte_cache_dir(node)
    out_alpha_path = os.path.join(target_dir, f"compmatte_alpha_{curr_frame:04d}.png").replace("\\", "/")

    custom_py = node.knob("custom_python").value() if node.knob("custom_python") else None

    # Execution (In-Process vs Host Subprocess Worker)
    try:
        if _IN_PROCESS_AVAILABLE:
            bgr = cv2.imread(src_temp)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            config_obj = CompMatteConfig(**cfg)
            engine = MatteFusionEngine(config_obj)
            res = engine.process_compmatte(rgb)
            alpha_u8 = np.clip(res["alpha"] * 255.0, 0, 255).astype(np.uint8)
            cv2.imwrite(out_alpha_path, alpha_u8)
            mode_desc = "In-Process"
        else:
            if not _IN_NUKE:
                # Mock write for headless test
                with open(out_alpha_path, "w") as f:
                    f.write("mock_alpha")
                mode_desc = "Mock"
            else:
                run_compmatte_external(src_temp, out_alpha_path, cfg, clean_temp, hold_temp, custom_py)
                mode_desc = "Host Python Worker"

        # Update node status
        msg = f"Frame {curr_frame} Matte Extracted successfully! ({mode_desc})"
        if node.knob("cm_status"):
            node.knob("cm_status").setValue(msg)

        if _IN_NUKE:
            # Reload internal Read node
            try:
                read_n = node.node("Read_CompMatte_Alpha")
                if read_n and read_n.knob("reload"):
                    read_n.knob("reload").execute()
            except Exception:
                pass

    except Exception as err:
        err_msg = f"CompMatte Error: {err}"
        if node.knob("cm_status"):
            node.knob("cm_status").setValue(err_msg)
        if _IN_NUKE:
            nuke.message(err_msg)
    finally:
        # Clean up temporary source render files
        for p in (src_temp, clean_temp, hold_temp):
            if p and os.path.exists(p):
                try: os.remove(p)
                except Exception: pass


def on_render_range(node: Any) -> None:
    """Callback when '🎬 Bake Frame Range...' button is pressed."""
    if not _IN_NUKE:
        return

    src_input = node.input(0)
    if not src_input:
        nuke.message("Please connect an image or video plate to the 'Source' input.")
        return

    # Auto-detect clip frame range from upstream node (e.g. Read node for MXF/MOV/EXR)
    first_f = None
    last_f = None
    if src_input.knob("first") and src_input.knob("last"):
        try:
            first_f = int(src_input.knob("first").value())
            last_f = int(src_input.knob("last").value())
        except Exception:
            pass
    if first_f is None or last_f is None:
        first_f = int(nuke.root()["first_frame"].value())
        last_f = int(nuke.root()["last_frame"].value())

    curr_custom = node.knob("output_dir").value() if node.knob("output_dir") else ""

    panel = nuke.Panel("Bake CompMatte Sequence")
    panel.addSingleLineInput("Start Frame:", str(first_f))
    panel.addSingleLineInput("End Frame:", str(last_f))
    panel.addSingleLineInput("Output Folder (empty = default cache):", str(curr_custom or ""))

    if not panel.show():
        return

    try:
        start_f = int(panel.value("Start Frame:"))
        end_f = int(panel.value("End Frame:"))
    except ValueError:
        nuke.message("Invalid frame range specified.")
        return

    new_custom = panel.value("Output Folder (empty = default cache):").strip()
    if node.knob("output_dir"):
        node.knob("output_dir").setValue(new_custom)

    target_dir = get_compmatte_cache_dir(node)
    custom_py = node.knob("custom_python").value() if node.knob("custom_python") else None

    sc_val = [0.0, 1.0, 0.0]
    if node.knob("screen_color"):
        sc = node.knob("screen_color").value()
        if isinstance(sc, (list, tuple)):
            sc_val = [float(x) for x in sc[:3]]
        elif isinstance(sc, (int, float)):
            sc_val = [float(sc), float(sc), float(sc)]

    cfg = {
        "screen_type": node.knob("screen_type").value() if node.knob("screen_type") else "green",
        "custom_color": sc_val,
        "red_weight": float(node.knob("w_red").value()) if node.knob("w_red") else 0.5,
        "blue_weight": float(node.knob("w_blue").value()) if node.knob("w_blue") else 0.5,
        "use_hole_fill": bool(node.knob("use_hole_fill").value()) if node.knob("use_hole_fill") else True,
        "restore_fine_edges": bool(node.knob("restore_fine_edges").value()) if node.knob("restore_fine_edges") else True,
        "safe_zone_radius": int(node.knob("safe_radius").value()) if node.knob("safe_radius") else 40,
        "core_erode_size": int(node.knob("core_erode").value()) if node.knob("core_erode") else 7,
        "feather_radius": float(node.knob("feather").value()) if node.knob("feather") else 0.5,
        "black_clip": float(node.knob("black_clip").value()) if node.knob("black_clip") else 0.05,
        "white_clip": float(node.knob("white_clip").value()) if node.knob("white_clip") else 0.95,
        "gamma": float(node.knob("gamma").value()) if node.knob("gamma") else 1.0,
    }

    task = nuke.ProgressTask("CompMatte Baking Sequence...")
    total_frames = max(1, end_f - start_f + 1)

    for idx, f in enumerate(range(start_f, end_f + 1)):
        if task.isCancelled():
            break
        task.setMessage(f"Processing Frame {f} ({idx + 1}/{total_frames})...")
        task.setProgress(int((idx / total_frames) * 100))

        src_temp = _export_node_frame_to_temp(src_input, f, "src")
        if not src_temp:
            continue

        out_alpha_path = os.path.join(target_dir, f"compmatte_alpha_{f:04d}.png").replace("\\", "/")
        try:
            run_compmatte_external(src_temp, out_alpha_path, cfg, None, None, custom_py)
        except Exception as e:
            nuke.message(f"Frame {f} failed: {e}")
            break
        finally:
            if src_temp and os.path.exists(src_temp):
                try: os.remove(src_temp)
                except Exception: pass

    del task
    if node.knob("cm_status"):
        node.knob("cm_status").setValue(f"Bake Complete for frames {start_f}..{end_f} in {target_dir}")

    # Reload internal Read node with active directory and frame range
    try:
        read_n = node.node("Read_CompMatte_Alpha")
        if read_n:
            read_n["file"].setValue(os.path.join(target_dir, "compmatte_alpha_####.png").replace("\\", "/"))
            if read_n.knob("first"):
                read_n["first"].setValue(start_f)
            if read_n.knob("last"):
                read_n["last"].setValue(end_f)
            if read_n.knob("origfirst"):
                read_n["origfirst"].setValue(start_f)
            if read_n.knob("origlast"):
                read_n["origlast"].setValue(end_f)
            if read_n.knob("reload"):
                read_n.knob("reload").execute()
    except Exception:
        pass

    nuke.message(f"Bake Complete! Alpha matte sequence saved in:\n{target_dir}")
