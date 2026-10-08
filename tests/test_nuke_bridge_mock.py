"""
tests/test_nuke_bridge_mock.py
Headless unit tests for compmatte_bridge outside active Nuke sessions.
"""

import os
import sys
import unittest

_pkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

import compmatte_bridge


class TestNukeBridgeMock(unittest.TestCase):

    def test_create_node(self):
        node = compmatte_bridge.create_compmatte_node()
        self.assertIsNotNone(node)
        self.assertEqual(node.Class(), "Group")
        
        knobs = node.knobs()
        self.assertIn("screen_type", knobs)
        self.assertIn("view_mode", knobs)
        self.assertIn("w_red", knobs)
        self.assertIn("btn_extract", knobs)
        self.assertIn("btn_range", knobs)
        self.assertIn("use_hole_fill", knobs)
        self.assertIn("restore_fine_edges", knobs)
        self.assertIn("safe_radius", knobs)
        self.assertIn("black_clip", knobs)
        self.assertIn("white_clip", knobs)

    def test_knob_defaults(self):
        node = compmatte_bridge.create_compmatte_node()
        self.assertEqual(node.knob("screen_type").value(), "green")
        self.assertEqual(node.knob("w_red").value(), 0.5)
        self.assertEqual(node.knob("use_hole_fill").value(), True)
        self.assertEqual(node.knob("restore_fine_edges").value(), True)
        self.assertEqual(node.knob("safe_radius").value(), 40)
        self.assertEqual(node.knob("black_clip").value(), 0.05)
        self.assertEqual(node.knob("white_clip").value(), 0.95)

    def test_extract_matte_mock(self):
        node = compmatte_bridge.create_compmatte_node()
        compmatte_bridge.on_extract_matte(node)
        status_knob = node.knob("cm_status")
        self.assertTrue("Extracted successfully" in status_knob.value())

    def test_environment_discovery(self):
        env_info = compmatte_bridge.check_environment()
        self.assertIn("available", env_info)
        self.assertIn("mode", env_info)
        self.assertTrue(env_info["available"])
        self.assertIsNotNone(env_info["numpy_version"])

    def test_find_compmatte_python(self):
        py_exe = compmatte_bridge.find_compmatte_python()
        self.assertIsNotNone(py_exe)
        self.assertTrue(os.path.exists(py_exe))

    def test_custom_output_directory(self):
        import tempfile
        node = compmatte_bridge.create_compmatte_node()
        self.assertIn("output_dir", node.knobs())
        self.assertIn("btn_open", node.knobs())

        # Test default fallback when output_dir is empty
        default_dir = compmatte_bridge.get_compmatte_cache_dir(node)
        self.assertTrue(os.path.isdir(default_dir))

        # Test custom directory override
        temp_custom = tempfile.mkdtemp(prefix="test_custom_out_")
        try:
            node.knob("output_dir").setValue(temp_custom)
            res_dir = compmatte_bridge.get_compmatte_cache_dir(node)
            self.assertEqual(os.path.normpath(res_dir), os.path.normpath(temp_custom))
        finally:
            import shutil
            shutil.rmtree(temp_custom, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
