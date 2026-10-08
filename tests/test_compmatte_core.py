"""
tests/test_compmatte_core.py
Synthetic unit tests for pure optical CompMatte for Nuke core engine.
"""

import os
import sys
import unittest
import numpy as np

_pkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

import compmatte_core
from compmatte_core import (
    CompMatteConfig,
    IBKEngine,
    CoreEngine,
    MatteFusionEngine,
    ScreenType,
)


class TestCompMatteCore(unittest.TestCase):

    def setUp(self):
        self.h, self.w = 200, 200
        self.frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        y, x = np.mgrid[0:self.h, 0:self.w]
        self.frame[:, :, 0] = 20
        self.frame[:, :, 1] = 210
        self.frame[:, :, 2] = 25

        center_y, center_x = 100, 100
        radius = 40
        dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        subject_mask = dist <= radius
        hole_mask = (np.sqrt((x - 100) ** 2 + (y - 95) ** 2) <= 10)

        self.frame[subject_mask, 0] = 160
        self.frame[subject_mask, 1] = 90
        self.frame[subject_mask, 2] = 70
        self.frame[hole_mask, 1] = 180

        for i in range(15):
            hx = 100 + radius + i
            hy = 100 + (i % 5) - 2
            if 0 <= hx < self.w and 0 <= hy < self.h:
                self.frame[hy, hx] = [60, 80, 50]

    def test_screen_difference(self):
        engine = IBKEngine(CompMatteConfig(screen_type="green"))
        diff = engine.compute_screen_difference(self.frame)
        self.assertEqual(diff.shape, (self.h, self.w))
        self.assertGreater(diff[10, 10], 0.3)
        self.assertLess(diff[120, 100], 0.0)

    def test_clean_plate_generation(self):
        engine = IBKEngine(CompMatteConfig(screen_type="green"))
        clean = engine.generate_clean_plate(self.frame, iterations=2)
        self.assertEqual(clean.shape, (self.h, self.w, 3))
        center_pixel = clean[100, 100]
        self.assertGreater(center_pixel[1], center_pixel[0])

    def test_ibk_matte_pulling(self):
        engine = IBKEngine(CompMatteConfig(screen_type="green"))
        alpha, clean = engine.pull_matte(self.frame)
        self.assertEqual(alpha.shape, (self.h, self.w))
        self.assertGreaterEqual(alpha.min(), 0.0)
        self.assertLessEqual(alpha.max(), 1.0)
        self.assertLess(alpha[10, 10], 0.1)

    def test_topological_hole_filling(self):
        core_engine = CoreEngine(CompMatteConfig(use_hole_fill=True))
        donut = np.zeros((100, 100), dtype=np.float32)
        y, x = np.mgrid[0:100, 0:100]
        donut[np.sqrt((x - 50)**2 + (y - 50)**2) <= 30] = 1.0
        donut[np.sqrt((x - 50)**2 + (y - 50)**2) <= 10] = 0.0

        filled = core_engine.fill_topological_holes(donut)
        self.assertEqual(filled[50, 50], 1.0)
        self.assertEqual(filled[5, 5], 0.0)

    def test_safe_zone_edge_re_injection(self):
        fusion_engine = MatteFusionEngine(CompMatteConfig(restore_fine_edges=True, safe_zone_radius=20))
        base = np.zeros((100, 100), dtype=np.float32)
        base[30:70, 30:70] = 1.0
        base[50, 75] = 0.1

        raw_edge = np.zeros((100, 100), dtype=np.float32)
        raw_edge[50, 75] = 0.85

        core = np.zeros((100, 100), dtype=np.float32)
        core[35:65, 35:65] = 1.0

        restored = fusion_engine.reinject_safe_zone_details(base, raw_edge, core, safe_radius=20)
        self.assertAlmostEqual(restored[50, 75], 0.85, places=2)

    def test_pure_value_locking(self):
        core_engine = CoreEngine(CompMatteConfig(black_clip=0.02, white_clip=0.98))
        noisy_alpha = np.array([0.005, 0.015, 0.5, 0.985, 0.995], dtype=np.float32)
        clamped = core_engine.clamp_pure_values(noisy_alpha)
        self.assertEqual(clamped[0], 0.0)
        self.assertEqual(clamped[1], 0.0)
        self.assertEqual(clamped[3], 1.0)
        self.assertEqual(clamped[4], 1.0)

    def test_full_pipeline_process(self):
        config = CompMatteConfig(screen_type="green", use_hole_fill=True, restore_fine_edges=True)
        fusion_engine = MatteFusionEngine(config)
        result = fusion_engine.process_compmatte(self.frame)

        self.assertIn("alpha", result)
        self.assertIn("clean_plate", result)
        self.assertIn("core_matte", result)
        self.assertIn("edge_matte", result)
        self.assertIn("premultiplied_rgb", result)

        alpha = result["alpha"]
        self.assertEqual(alpha.shape, (self.h, self.w))
        self.assertEqual(alpha[5, 5], 0.0)
        self.assertEqual(alpha[120, 100], 1.0)


if __name__ == "__main__":
    unittest.main()
