from unittest import TestCase

import numpy as np

from eyetracker.geometry import Ray, calibrate, closest_approach, iris_direction


def toward(origin, point):
    delta = point - origin
    return Ray(origin, delta / np.linalg.norm(delta))


class RayTests(TestCase):
    def test_skew_lines_return_midpoint_and_gap(self):
        first = Ray(np.array([-2.0, 0, 4]), np.array([1, 0, -1]) / np.sqrt(2))
        second = Ray(np.array([2.0, 6, 4]), np.array([-1, 0, -1]) / np.sqrt(2))
        estimate = closest_approach(first, second)
        np.testing.assert_allclose(estimate.midpoint, [0, 3, 2])
        self.assertAlmostEqual(estimate.gap, 6)

    def test_parallel_or_oppositely_directed_rays_do_not_converge_forward(self):
        first = Ray(np.array([-1.0, 0, 4]), np.array([0.0, 0, -1]))
        second = Ray(np.array([1.0, 0, 4]), np.array([0.0, 0, -1]))
        self.assertIsNone(closest_approach(first, second))
        self.assertIsNone(closest_approach(
            toward(first.origin, np.zeros(3)),
            Ray(second.origin, -toward(second.origin, np.zeros(3)).direction),
        ))

    def test_iris_uses_camera_facing_surface(self):
        np.testing.assert_allclose(
            iris_direction(np.array([0, 0, 500]), np.array([0, 0, 1])),
            [0, 0, -1],
        )


class CalibrationTests(TestCase):
    center = np.array([15, 190, -50])
    right = np.array([12, 0, 1]) * (490 / np.hypot(12, 1))
    down = np.array([0, 275, 0])
    origins = (np.array([-32, 0, 550]), np.array([32, 0, 550]))

    def screen_point(self, target):
        u, v = target
        return self.center + (u - 0.5) * self.right + (v - 0.5) * self.down

    def samples(self, make_rays):
        targets = [(u, v) for u in (0.1, 0.5, 0.9) for v in (0.1, 0.5, 0.9)]
        return [(target, make_rays(self.screen_point(target))) for target in targets]

    def test_recovers_3d_screen_and_predicts_unseen_point(self):
        rng = np.random.default_rng(42)

        def frames(point):
            return [
                tuple(toward(eye, point + rng.normal(0, 1.5, size=3)) for eye in self.origins)
                for _ in range(14)
            ] + [tuple(toward(eye, point + [30, -65, -140]) for eye in self.origins)]

        calibration = calibrate(self.samples(frames), 490 / 275)
        self.assertIsNotNone(calibration.plane)
        np.testing.assert_allclose(calibration.plane.center, self.center, atol=5)
        np.testing.assert_allclose(calibration.plane.right, self.right, atol=8)
        np.testing.assert_allclose(calibration.plane.down, self.down, atol=8)
        target = (0.73, 0.29)
        # Le visage se déplace légèrement après la calibration.
        rays = tuple(toward(eye + [20, 5, 15], self.screen_point(target)) for eye in self.origins)
        np.testing.assert_allclose(calibration.predict(rays), target, atol=0.015)

    def test_parallel_rays_can_still_provide_screen_coordinates(self):
        def frames(point):
            direction = toward(np.mean(self.origins, axis=0), point).direction
            return [(Ray(self.origins[0], direction), Ray(self.origins[1], direction))] * 14

        calibration = calibrate(self.samples(frames), 490 / 275)
        self.assertIsNone(calibration.plane)
        np.testing.assert_allclose(calibration.predict(frames(self.center)[0]), [0.5, 0.5], atol=0.02)

    def test_rejects_calibration_with_motionless_gaze(self):
        fixed = tuple(toward(eye, self.center) for eye in self.origins)
        with self.assertRaises(ValueError):
            calibrate(self.samples(lambda _: [fixed] * 14), 490 / 275)
