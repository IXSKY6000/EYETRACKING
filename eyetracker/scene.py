"""Vue orbitale des mesures de suivi dans le repère de la webcam."""

from math import cos, sin
import tkinter as tk
from tkinter import ttk

import mediapipe as mp
import numpy as np

from .geometry import Ray, ScreenPlane, closest_approach
from .vision import Observation


FACE_EDGES = np.array(sorted(mp.solutions.face_mesh.FACEMESH_CONTOURS))
RAY_COLORS = ("#7fdcb9", "#f6b77e")
SCREEN_COLOR = "#77aef9"
SCREEN_CORNERS = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])


def screen_points(plane: ScreenPlane, coordinates: np.ndarray) -> np.ndarray:
    return plane.center + coordinates @ np.stack((plane.right, plane.down))


def project_view(points: np.ndarray, center: np.ndarray, yaw: float, pitch: float) -> np.ndarray:
    """Projette dans le repère de la vue orthographique orbitale."""
    yaw_matrix = np.array([[cos(yaw), 0, sin(yaw)], [0, 1, 0], [-sin(yaw), 0, cos(yaw)]])
    pitch_matrix = np.array([[1, 0, 0], [0, cos(pitch), -sin(pitch)], [0, sin(pitch), cos(pitch)]])
    return (points - center) @ yaw_matrix.T @ pitch_matrix.T


def ray_endpoint(ray: Ray, plane: ScreenPlane | None, default_length: float) -> tuple[np.ndarray, bool]:
    if plane is not None:
        hit = plane.project(ray)
        if hit is not None and np.max(np.abs(hit)) <= 0.5:
            return screen_points(plane, hit), True
    return ray.origin + default_length * ray.direction, False


class SceneView:
    def __init__(self, root: tk.Tk):
        self.window = tk.Toplevel(root, bg="#101718")
        self.window.title("Eye Tracker · Vue 3D")
        self.window.geometry("880x640")
        self.window.minsize(630, 440)
        header = tk.Frame(self.window, bg="#101718")
        header.pack(fill="x", padx=18, pady=12)
        tk.Label(header, text="Vue 3D", bg="#101718", fg="#eff5f0",
                 font=("TkDefaultFont", 17, "bold")).pack(side="left")
        ttk.Button(header, text="Recentrer", command=self.reset).pack(side="right")
        self.canvas = tk.Canvas(self.window, bg="#0a1115", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=18)
        footer = tk.Frame(self.window, bg="#101718")
        footer.pack(fill="x", padx=18, pady=10)
        for label, color in (
            ("Écran", SCREEN_COLOR), ("Rayon œil 1", RAY_COLORS[0]),
            ("Rayon œil 2", RAY_COLORS[1]), ("Convergence", "#ffe08c"),
        ):
            tk.Label(footer, text=label, fg=color, bg="#101718").pack(side="left", padx=(0, 12))
        tk.Label(footer, text="Glisser pour tourner · Molette pour zoomer", fg="#a6b8b3",
                 bg="#101718").pack(side="right")
        self.status = tk.StringVar()
        tk.Label(self.window, textvariable=self.status, fg="#a6b8b3", bg="#101718",
                 anchor="w").pack(fill="x", padx=18, pady=(0, 10))
        self.face: Observation | None = None
        self.plane: ScreenPlane | None = None
        self.anchor: np.ndarray | None = None
        self.radius = 350.0
        self.drag = (0, 0)
        self.canvas.bind("<Configure>", lambda _event: self.render())
        self.canvas.bind("<ButtonPress-1>", self.begin_orbit)
        self.canvas.bind("<B1-Motion>", self.orbit)
        self.canvas.bind("<MouseWheel>", self.wheel)
        self.canvas.bind("<Button-4>", lambda _event: self.change_zoom(1))
        self.canvas.bind("<Button-5>", lambda _event: self.change_zoom(-1))
        self.reset()

    def reset(self) -> None:
        self.yaw, self.pitch, self.zoom = -0.77, -0.38, 1.0
        self.anchor = None
        self.render()

    def begin_orbit(self, event) -> None:
        self.drag = (event.x, event.y)

    def orbit(self, event) -> None:
        x, y = self.drag
        self.yaw += (event.x - x) * 0.009
        self.pitch = float(np.clip(self.pitch + (event.y - y) * 0.009, -1.5, 1.5))
        self.drag = (event.x, event.y)
        self.render()

    def wheel(self, event) -> None:
        if event.delta:
            self.change_zoom(1 if event.delta > 0 else -1)

    def change_zoom(self, direction: int) -> None:
        self.zoom = float(np.clip(self.zoom * 1.14**direction, 0.45, 5))
        self.render()

    def update(self, face: Observation | None, plane: ScreenPlane | None) -> None:
        if face is self.face and plane is self.plane:
            return
        if plane is not self.plane:
            self.anchor = None
        self.face, self.plane = face, plane
        self.render()

    def render(self) -> None:
        canvas = self.canvas
        canvas.delete("all")
        face, plane = self.face, self.plane
        corners = screen_points(plane, SCREEN_CORNERS) if plane is not None else None
        bounds = [np.zeros((1, 3))]
        if corners is not None:
            bounds.append(corners)
        if face is not None:
            bounds.append(face.landmarks)
        if self.anchor is None and len(bounds) > 1:
            points = np.concatenate(bounds)
            self.anchor = (points.min(axis=0) + points.max(axis=0)) / 2
            self.radius = max(130, float(np.max(np.linalg.norm(points - self.anchor, axis=1))))
        center = self.anchor if self.anchor is not None else np.zeros(3)
        width, height = canvas.winfo_width(), canvas.winfo_height()
        scale = self.zoom * min(width, height) * 0.44 / self.radius

        def projected(points):
            coords = project_view(points, center, self.yaw, self.pitch)
            return coords[:, :2] * scale + [width / 2, height / 2]

        def path(points, **options):
            canvas.create_line(*projected(points).ravel(), **options)

        def dot(point, radius, color):
            x, y = projected(point[None, :])[0]
            canvas.create_oval(x-radius, y-radius, x+radius, y+radius, fill=color, outline="")

        if corners is not None:
            canvas.create_polygon(*projected(corners).ravel(), fill="#172943", outline=SCREEN_COLOR, width=2)
            for offset in (-0.25, 0, 0.25):
                path(screen_points(plane, np.array([[offset, -0.5], [offset, 0.5]])), fill="#375478")
                path(screen_points(plane, np.array([[-0.5, offset], [0.5, offset]])), fill="#375478")
            x, y = projected(corners[:1])[0]
            canvas.create_text(x, y-11, text="Écran estimé", anchor="sw", fill=SCREEN_COLOR)

        origin = np.zeros(3)
        for axis, label, color in zip(np.eye(3), "XYZ", ("#e78c90", "#82bd91", "#83a9f2")):
            end = axis * 55
            path(np.stack((origin, end)), fill=color, arrow="last")
            x, y = projected(end[None, :])[0]
            canvas.create_text(x+6, y-6, text=label, fill=color)
        dot(origin, 3, "#eff5f0")
        x, y = projected(origin[None, :])[0]
        canvas.create_text(x-8, y+12, anchor="e", text="Webcam", fill="#a6b8b3")

        status = ["Écran 3D estimé" if plane is not None else "Calibrez l’écran pour afficher son plan 3D."]
        if face is None:
            status.append("Visage en attente.")
        else:
            outline = projected(face.landmarks)[FACE_EDGES]
            for edge in outline:
                canvas.create_line(*edge.ravel(), fill="#b3c7db", width=1.4)
            for eye, color in zip(face.eyes, RAY_COLORS):
                dot(eye, 4, color)
            if face.rays is not None:
                self.draw_gaze(face, plane, path, dot, status)
            else:
                status.append("Deux yeux ouverts nécessaires pour afficher les rayons.")
        self.status.set("  ·  ".join(status))

    @staticmethod
    def draw_gaze(face, plane, path, dot, status) -> None:
        convergence = closest_approach(*face.rays)
        length = float(np.clip(face.face[2], 150, 1400))
        if convergence is not None:
            distance = np.linalg.norm(convergence.midpoint - face.face)
            if distance < length * 1.8 and convergence.gap <= 50:
                dot(convergence.midpoint, 5, "#ffe08c")
            status.append(f"Écart entre les rayons {convergence.gap:.1f} mm")
        for ray, color in zip(face.rays, RAY_COLORS):
            endpoint, on_screen = ray_endpoint(ray, plane, length)
            path(np.stack((ray.origin, endpoint)), fill=color, width=2, arrow="last")
            if on_screen:
                dot(endpoint, 4, color)
