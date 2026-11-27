from unittest import TestCase

from eyetracker.gestures import BrowClicks


class BrowTests(TestCase):
    def setUp(self):
        self.clicks = BrowClicks()

    def update(self, timestamp, lift):
        return self.clicks.update(timestamp, lift, 0.08)

    def test_single_lift_waits_for_double_window_and_does_not_repeat(self):
        self.assertIsNone(self.update(0, 0))
        self.assertIsNone(self.update(0.1, 0.14))
        self.assertIsNone(self.update(0.19, 0.14))
        self.assertIsNone(self.update(0.6, 0.14))
        self.assertEqual(self.update(0.8, 0.14), "left")
        self.assertIsNone(self.update(1.5, 0.14))

    def test_double_lift_emits_only_right_click(self):
        outcomes = [
            self.update(time, lift) for time, lift in [
                (0, 0), (0.1, 0.13), (0.2, 0.13), (0.3, 0.01),
                (0.68, 0.14), (0.73, 0.14), (0.77, 0.14), (1.2, 0),
            ]
        ]
        self.assertEqual([result for result in outcomes if result], ["right"])

    def test_reset_cancels_pending_click_and_requires_neutral(self):
        self.update(0, 0)
        self.update(0.1, 0.14)
        self.update(0.2, 0.14)
        self.clicks.reset()
        self.assertIsNone(self.update(0.9, 0.14))
        self.assertIsNone(self.update(1.6, 0.14))
        self.assertIsNone(self.update(2, 0))

    def test_brief_spike_does_not_click(self):
        self.update(0, 0)
        self.update(0.1, 0.12)
        self.update(0.13, 0.09)
        self.update(0.15, 0)
        self.assertIsNone(self.update(1, 0))
