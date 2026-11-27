"""Détection des levées de sourcils avec hystérésis et délai de double geste."""

from typing import Literal

Click = Literal["left", "right"]


class BrowClicks:
    def __init__(self, double_window: float = 0.65, minimum_hold: float = 0.075):
        self.double_window = double_window
        self.minimum_hold = minimum_hold
        self.reset()

    def reset(self) -> None:
        self._armed = False
        self._lift_started: float | None = None
        self._first_lift: float | None = None

    def update(self, now: float, lift: float, threshold: float) -> Click | None:
        # Un retour à la position neutre arme le geste, y compris après une perte du visage.
        if lift <= threshold * 0.55:
            self._armed = True
            self._lift_started = None
        elif self._armed and lift >= threshold:
            if self._lift_started is None:
                self._lift_started = now
            if now - self._lift_started >= self.minimum_hold:
                onset = self._lift_started
                self._armed = False
                self._lift_started = None
                previous = self._first_lift
                self._first_lift = onset
                if previous is not None:
                    if onset - previous <= self.double_window:
                        self._first_lift = None
                        return "right"
                    return "left"
        elif self._armed:
            self._lift_started = None

        if self._first_lift is not None and now - self._first_lift > self.double_window:
            if self._lift_started is None or self._lift_started - self._first_lift > self.double_window:
                self._first_lift = None
                return "left"
        return None
