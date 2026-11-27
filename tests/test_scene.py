from unittest import TestCase

import numpy as np

from eyetracker.geometry import Ray, ScreenPlane
from eyetracker.scene import project_view, ray_endpoint, screen_points


class SceneTests(TestCase):
    plane = ScreenPlane(np.array([0, 90, 0]), np.array([400, 0, 0]), np.array([0, 240, 0]))

    def test_screen_corners_follow_calibrated_plane(self):
        corners = screen_points(self.plane, np.array([[-0.5, -0.5], [0.5, 0.5]]))
        np.testing.assert_allclose(corners, [[-200, -30, 0], [200, 210, 0]])

    def test_gaze_hit_and_miss(self):
        ray = Ray(np.array([15, 50, 500]), np.array([0, 0, -1]))
        endpoint, on_screen = ray_endpoint(ray, self.plane, 250)
        self.assertTrue(on_screen)
        np.testing.assert_allclose(endpoint, [15, 50, 0])
        outside = Ray(np.array([250, 50, 500]), ray.direction)
        endpoint, on_screen = ray_endpoint(outside, self.plane, 250)
        self.assertFalse(on_screen)
        np.testing.assert_allclose(endpoint, [250, 50, 250])

    def test_orbit_rotates_depth_into_visible_x(self):
        center = np.array([30, 40, 50])
        points = center + np.array([[0, 0, 10]])
        np.testing.assert_allclose(project_view(points, center, np.pi / 2, 0), [[10, 0, 0]], atol=1e-8)
