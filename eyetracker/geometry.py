"""Géométrie dans le repère caméra OpenCV, en millimètres estimés."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Ray:
    origin: np.ndarray
    direction: np.ndarray  # Vecteur unitaire pointant de l'œil vers l'écran.


@dataclass(frozen=True)
class Convergence:
    midpoint: np.ndarray
    gap: float


def closest_approach(first: Ray, second: Ray) -> Convergence | None:
    """Milieu du plus petit segment entre deux demi-droites orientées."""
    cosine = first.direction @ second.direction
    denominator = 1 - cosine * cosine
    if denominator < 1e-5:
        return None

    offset = second.origin - first.origin
    first_projection = offset @ first.direction
    second_projection = offset @ second.direction
    first_distance = (first_projection - cosine * second_projection) / denominator
    second_distance = (cosine * first_projection - second_projection) / denominator
    if min(first_distance, second_distance) <= 0:
        return None

    a = first.origin + first_distance * first.direction
    b = second.origin + second_distance * second.direction
    return Convergence((a + b) / 2, float(np.linalg.norm(a - b)))


@dataclass(frozen=True)
class ScreenPlane:
    center: np.ndarray
    right: np.ndarray
    down: np.ndarray

    def project(self, ray: Ray) -> np.ndarray | None:
        normal = np.cross(self.right, self.down)
        denominator = normal @ ray.direction
        if abs(denominator) < 0.1 * np.linalg.norm(normal):
            return None
        distance = (normal @ (self.center - ray.origin)) / denominator
        if distance <= 0:
            return None
        offset = ray.origin + distance * ray.direction - self.center
        return np.array([
            offset @ self.right / (self.right @ self.right),
            offset @ self.down / (self.down @ self.down),
        ])


def fit_screen_plane(
    targets: np.ndarray, points: np.ndarray, weights: np.ndarray, aspect: float
) -> ScreenPlane:
    """Ajuste un rectangle 3D aux points de convergence du regard."""
    coordinates = (targets - 0.5) * [aspect, 1]
    base_weights = weights / weights.sum()
    for _ in range(6):
        normalized = weights / weights.sum()
        origin = normalized @ coordinates
        center = normalized @ points
        source = coordinates - origin
        destination = points - center
        left, singular, right = np.linalg.svd(
            (source.T * normalized) @ destination, full_matrices=False
        )
        axes = left @ right
        height = singular.sum() / (normalized @ np.sum(source**2, axis=1))
        fitted_center = center - height * origin @ axes
        residuals = np.linalg.norm(points - fitted_center - height * coordinates @ axes, axis=1)
        weights = base_weights / np.maximum(1, residuals / max(12, height * 0.12))

    if not 65 <= height <= 950 or np.median(residuals) > height * 0.30:
        raise ValueError("Les points de convergence ne forment pas un écran cohérent.")
    if abs(np.cross(*axes)[2]) < 0.25:
        raise ValueError("Le plan estimé est presque de profil par rapport à la caméra.")
    return ScreenPlane(fitted_center, height * aspect * axes[0], height * axes[1])


@dataclass(frozen=True)
class Calibration:
    plane: ScreenPlane | None
    center: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    rms_error: float

    @staticmethod
    def features(rays: tuple[Ray, Ray], plane: ScreenPlane | None) -> np.ndarray | None:
        if plane is not None:
            points = [plane.project(ray) for ray in rays]
            if all(point is not None for point in points):
                return np.mean(points, axis=0)
            return None

        # Intersection avec le plan de la caméra, utilisé comme repère 2D.
        if any(ray.direction[2] >= -0.2 for ray in rays):
            return None
        points = [
            (ray.origin[:2] - ray.origin[2] / ray.direction[2] * ray.direction[:2]) / 500
            for ray in rays
        ]
        return np.mean(points, axis=0)

    def predict(self, rays: tuple[Ray, Ray]) -> np.ndarray | None:
        feature = self.features(rays, self.plane)
        if feature is None:
            return None
        return np.r_[1, (feature - self.center) / self.scale] @ self.coefficients


def calibrate(
    samples: list[tuple[tuple[float, float], list[tuple[Ray, Ray]]]], aspect: float
) -> Calibration:
    targets = np.asarray([target for target, _ in samples])
    plane_targets, midpoints, weights = [], [], []
    for target, frames in samples:
        estimates = []
        for rays in frames:
            estimate = closest_approach(*rays)
            if estimate is None:
                continue
            eye_distance = (rays[0].origin[2] + rays[1].origin[2]) / 2 - estimate.midpoint[2]
            if 120 <= eye_distance <= 2000 and estimate.gap <= 50:
                estimates.append(estimate)
        if len(estimates) < max(3, len(frames) // 3):
            continue
        plane_targets.append(target)
        midpoints.append(np.median([estimate.midpoint for estimate in estimates], axis=0))
        weights.append(1 / (5 + np.median([estimate.gap for estimate in estimates])))

    plane = None
    if len(midpoints) >= 6 and np.ptp(plane_targets, axis=0).min() > 0.4:
        try:
            plane = fit_screen_plane(
                np.asarray(plane_targets), np.asarray(midpoints), np.asarray(weights), aspect
            )
        except ValueError:
            pass

    def fit_mapping(candidate: ScreenPlane | None) -> Calibration:
        medians = []
        for _, frames in samples:
            features = [Calibration.features(rays, candidate) for rays in frames]
            usable = [feature for feature in features if feature is not None]
            if len(usable) < max(3, len(frames) // 2):
                raise ValueError("Regard instable sur un repère. Recommencez la calibration.")
            medians.append(np.median(usable, axis=0))

        medians = np.asarray(medians)
        center, scale = medians.mean(axis=0), medians.std(axis=0)
        if np.any(scale < 0.01):
            raise ValueError("Mouvement du regard insuffisant. Suivez chaque repère des yeux.")
        design = np.column_stack((np.ones(len(medians)), (medians - center) / scale))
        coefficients, _, rank, _ = np.linalg.lstsq(design, targets, rcond=None)
        error = float(np.sqrt(np.mean(np.sum((design @ coefficients - targets)**2, axis=1))))
        if rank < 3 or error > 0.19:
            raise ValueError("Calibration trop imprécise. Réessayez avec la tête stable et le visage éclairé.")
        return Calibration(candidate, center, scale, coefficients, error)

    if plane is not None:
        try:
            return fit_mapping(plane)
        except ValueError:
            pass
    return fit_mapping(None)


def iris_direction(eye_center: np.ndarray, camera_ray: np.ndarray, radius: float = 12) -> np.ndarray:
    """Place l'iris sur la partie visible d'un globe oculaire approximatif."""
    distance = camera_ray @ eye_center
    nearest = distance * camera_ray - eye_center
    depth = np.sqrt(max(0, radius**2 - nearest @ nearest))
    axis = nearest - depth * camera_ray
    return axis / np.linalg.norm(axis)
