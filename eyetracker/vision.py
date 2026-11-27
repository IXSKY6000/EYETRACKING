"""Repères Face Mesh, pose de la tête et estimation des deux rayons du regard."""

from dataclasses import dataclass
from math import radians, tan

import cv2
import mediapipe as mp
import numpy as np

from .geometry import Ray, iris_direction


# Extrait du modèle canonique MediaPipe (Google, Apache-2.0), converti
# des centimètres vers les millimètres et les axes de la caméra OpenCV.
POSE_POINTS = {
    1: (0, 11.27, -74.76),
    6: (0, -24.73, -57.89),
    33: (-44.46, -26.64, -31.73),
    133: (-18.56, -25.85, -37.58),
    168: (0, -32.71, -52.36),
    197: (0, -17.28, -63.17),
    263: (44.46, -26.64, -31.73),
    362: (18.56, -25.85, -37.58),
    127: (-77.43, -23.65, 20.05),
    356: (77.43, -23.65, 20.05),
}
MODEL = np.array(list(POSE_POINTS.values()), dtype=np.float64)
# Centres nominaux des globes oculaires dans le même repère.
EYE_CENTERS = np.array([[-31.77, -26.30, -26.30], [31.77, -26.30, -26.30]])
EYE_LANDMARKS = (
    (33, 133, 159, 145, 468, (63, 105, 66)),
    (362, 263, 386, 374, 473, (296, 334, 293)),
)


@dataclass(frozen=True)
class Observation:
    face: np.ndarray
    eyes: tuple[np.ndarray, np.ndarray]
    rays: tuple[Ray, Ray] | None
    brows: np.ndarray
    landmarks: np.ndarray


class FaceTracker:
    def __init__(self, horizontal_fov: float = 65):
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.fov = horizontal_fov
        self.frame_size: tuple[int, int] | None = None
        self.distortion = np.zeros(4)

    def close(self) -> None:
        self.mesh.close()

    def update(self, frame: np.ndarray) -> Observation | None:
        height, width = frame.shape[:2]
        if self.frame_size != (width, height):
            self.frame_size = (width, height)
            focal = width / (2 * tan(radians(self.fov) / 2))
            self.camera = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])
            self.inverse_camera = np.linalg.inv(self.camera)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None

        landmarks = np.array([(p.x * width, p.y * height, p.z * width)
                              for p in results.multi_face_landmarks[0].landmark])
        pixels = landmarks[:, :2]
        image_points = pixels[list(POSE_POINTS)]
        found, rotation, translation = cv2.solvePnP(
            MODEL, image_points, self.camera, self.distortion, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not found or not 120 < translation[2, 0] < 3500:
            return None

        transform, _ = cv2.Rodrigues(rotation)
        if transform[2, 2] < 0.55:
            return None
        reprojected, _ = cv2.projectPoints(MODEL, rotation, translation, self.camera, self.distortion)
        face_width = np.linalg.norm(pixels[127] - pixels[356])
        error = np.sqrt(np.mean(np.sum((reprojected[:, 0] - image_points)**2, axis=1)))
        if face_width < 45 or error > 0.09 * face_width:
            return None

        face = translation.ravel()
        eyes = tuple(EYE_CENTERS @ transform.T + face)
        rays, brows = [], []
        for eye, (left, right, upper, lower, iris, brow) in zip(eyes, EYE_LANDMARKS):
            side = pixels[right] - pixels[left]
            span = np.linalg.norm(side)
            if span < 5:
                return None
            down = np.array([-side[1], side[0]]) / span
            eye_line = (pixels[left] + pixels[right]) / 2
            brows.append(down @ (eye_line - pixels[list(brow)].mean(axis=0)) / span)
            opening = down @ (pixels[lower] - pixels[upper]) / span
            if opening < 0.10:
                continue

            camera_ray = self.inverse_camera @ np.r_[pixels[iris], 1]
            camera_ray /= np.linalg.norm(camera_ray)
            direction = iris_direction(eye, camera_ray)
            if direction[2] < -0.15:
                rays.append(Ray(eye, direction))

            cv2.circle(frame, tuple(pixels[iris].astype(int)), 3, (120, 255, 160), 1, cv2.LINE_AA)
            brow_line = pixels[list(brow)].astype(np.int32)
            cv2.polylines(frame, [brow_line], False, (100, 213, 255), 1, cv2.LINE_AA)

        anchor_depth = (transform @ POSE_POINTS[168] + face)[2]
        relative_depth = (landmarks[:, 2] - landmarks[168, 2]) * anchor_depth / self.camera[0, 0]
        pixel_rays = np.column_stack((pixels, np.ones(len(pixels)))) @ self.inverse_camera.T
        face_landmarks = pixel_rays * (anchor_depth + relative_depth)[:, None]
        return Observation(
            face, eyes, (rays[0], rays[1]) if len(rays) == 2 else None, np.array(brows), face_landmarks,
        )
