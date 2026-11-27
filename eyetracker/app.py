"""Application de bureau et calibration plein écran."""

import argparse
from dataclasses import dataclass
import os
import sys
from threading import Event, Thread
from time import monotonic
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from .geometry import Calibration, Ray, calibrate, closest_approach
from .gestures import BrowClicks
from .vision import FaceTracker, Observation

BACKGROUND = "#101718"
PANEL = "#182123"
TEXT = "#eff5f0"
MUTED = "#a6b8b3"
ACCENT = "#83efb4"
TARGETS = (
    (0.50, 0.50), (0.10, 0.13), (0.50, 0.13), (0.90, 0.13),
    (0.90, 0.50), (0.90, 0.87), (0.50, 0.87), (0.10, 0.87), (0.10, 0.50),
)


@dataclass(frozen=True)
class Frame:
    timestamp: float
    image: np.ndarray
    face: Observation | None


class Camera(Thread):
    def __init__(self, index: int, fov: float):
        super().__init__(daemon=True)
        self.index = index
        self.fov = fov
        self.stop = Event()
        self.latest: Frame | None = None
        self.error: str | None = None

    def run(self) -> None:
        capture = cv2.VideoCapture(self.index)
        tracker = None
        try:
            if not capture.isOpened():
                self.error = f"Caméra {self.index} indisponible. Vérifiez son branchement et les autorisations."
                return
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
            tracker = FaceTracker(self.fov)
            misses = 0
            while not self.stop.is_set():
                read, image = capture.read()
                timestamp = monotonic()
                if not read:
                    misses += 1
                    if misses == 10:
                        self.error = "La caméra ne transmet plus d’image."
                        return
                    self.stop.wait(0.05)
                    continue
                misses = 0
                self.latest = Frame(timestamp, image, tracker.update(image))
        except Exception as exc:
            self.error = f"Le suivi caméra s’est arrêté. {exc}"
        finally:
            capture.release()
            if tracker is not None:
                tracker.close()


class Application:
    def __init__(self, root: tk.Tk, camera: int, fov: float):
        self.root = root
        self.camera_index = camera
        self.fov = fov
        self.root.title("Eye Tracker")
        self.root.geometry("930x575")
        self.root.minsize(730, 500)
        self.root.configure(bg=BACKGROUND)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<Escape>", self.escape)

        self.calibration: Calibration | None = None
        self.brow_baseline: np.ndarray | None = None
        self.target_index: int | None = None
        self.mouse = None
        self.screen_size = (0, 0)
        self.calibrated_size = (0, 0)
        self.filtered: np.ndarray | None = None
        self.clicks = BrowClicks()
        self.last_frame: Frame | None = None
        self.last_preview = 0.0
        self.last_gaze = 0.0
        self.notice = "Commencez par calibrer l’écran."
        self.threshold = tk.DoubleVar(value=0.085)
        self.status = tk.StringVar(value="Ouverture de la caméra…")
        self.detail = tk.StringVar(value="En attente du visage")
        self.result = tk.StringVar(value="Écran à calibrer")
        self._layout()
        self.start_camera()
        self.root.after(25, self.tick)

    def _layout(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("Control.TButton", background=PANEL, foreground=TEXT, padding=(12, 9), borderwidth=1)
        style.configure("Accent.TButton", background=ACCENT, foreground=BACKGROUND, padding=(12, 9))
        style.map("Control.TButton", background=[("active", "#263836")], foreground=[("disabled", MUTED)])
        style.map("Accent.TButton", background=[("active", "#adf5ca"), ("disabled", PANEL)],
                  foreground=[("disabled", MUTED)])
        style.configure("Control.Horizontal.TScale", background=PANEL, troughcolor="#2c403e")

        header = tk.Frame(self.root, bg=BACKGROUND)
        header.pack(fill="x", padx=25, pady=(18, 14))
        tk.Label(header, text="Eye Tracker", bg=BACKGROUND, fg=TEXT, font=("TkDefaultFont", 24, "bold")).pack(anchor="w")
        tk.Label(header, textvariable=self.status, bg=BACKGROUND, fg=MUTED, font=("TkDefaultFont", 10),
                 wraplength=690, justify="left").pack(anchor="w", pady=(4, 0))

        body = tk.Frame(self.root, bg=BACKGROUND)
        body.pack(expand=True, fill="both", padx=25, pady=(0, 18))
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        self.preview = tk.Canvas(body, bg="#091012", highlightthickness=0, width=1, height=1)
        self.preview.grid(row=0, column=0, sticky="nsew", padx=(0, 15))
        self.preview_image = self.preview.create_image(0, 0)
        self.preview_label = self.preview.create_text(0, 0, text="Aperçu webcam", fill=MUTED)
        self.preview.bind("<Configure>", self.center_preview)

        controls = tk.Frame(body, bg=PANEL, padx=15, pady=14)
        controls.grid(row=0, column=1, sticky="nsew")
        tk.Label(controls, textvariable=self.result, bg=PANEL, fg=ACCENT, font=("TkDefaultFont", 11, "bold"),
                 wraplength=260, justify="left").pack(anchor="w")
        self.calibrate_button = ttk.Button(controls, text="Calibrer l’écran", style="Accent.TButton",
                                           command=self.begin_calibration)
        self.calibrate_button.pack(fill="x", pady=(10, 5))
        self.mouse_button = ttk.Button(controls, text="Activer le pointeur", style="Control.TButton",
                                       command=self.toggle_mouse, state="disabled")
        self.mouse_button.pack(fill="x")
        self.camera_button = ttk.Button(controls, text="Relancer la caméra", style="Control.TButton",
                                         command=self.start_camera)
        tk.Label(controls, text="Seuil de levée des sourcils", bg=PANEL, fg=TEXT).pack(anchor="w", pady=(13, 1))
        ttk.Scale(controls, from_=0.04, to=0.16, variable=self.threshold,
                  style="Control.Horizontal.TScale").pack(fill="x")
        tk.Label(controls, text="Une levée : clic gauche\nDeux levées rapprochées : clic droit", bg=PANEL, fg=MUTED,
                 justify="left").pack(anchor="w", pady=(6, 12))
        tk.Label(controls, textvariable=self.detail, bg=PANEL, fg=MUTED, justify="left",
                 font=("TkFixedFont", 9), wraplength=275).pack(anchor="w")

    def start_camera(self) -> None:
        self.camera_button.pack_forget()
        self.camera = Camera(self.camera_index, self.fov)
        self.camera.start()

    def begin_calibration(self) -> None:
        packet = self.camera.latest
        if packet is None or packet.face is None or packet.face.rays is None or monotonic() - packet.timestamp > 0.35:
            messagebox.showinfo("Calibration", "Placez votre visage face à la caméra jusqu’à ce que les deux yeux soient visibles.")
            return

        self.pause_mouse()
        self.target_index = -1
        self.neutral_samples: list[np.ndarray] = []
        self.samples: list[tuple[tuple[float, float], list[tuple[Ray, Ray]]]] = []
        self.cal_window = tk.Toplevel(self.root, bg=BACKGROUND)
        self.cal_window.title("Calibration du regard")
        width, height = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.cal_window.geometry(f"{width}x{height}+0+0")
        self.cal_window.attributes("-fullscreen", True)
        self.cal_window.protocol("WM_DELETE_WINDOW", self.end_calibration)
        self.cal_window.bind("<Escape>", self.escape)
        self.canvas = tk.Canvas(self.cal_window, bg=BACKGROUND, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _event: self.draw_target())
        self.cal_window.focus_force()
        self.target_since = monotonic()
        self.last_usable: float | None = None

    def draw_target(self, progress: float = 0, tracking: bool = True) -> None:
        if self.target_index is None:
            return
        self.canvas.delete("all")
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        neutral = self.target_index == -1
        u, v = (0.5, 0.5) if neutral else TARGETS[self.target_index]
        x, y = u * width, v * height
        color = ACCENT if tracking else MUTED
        self.canvas.create_text(width / 2, 28, text="CALIBRATION DU REGARD", fill=MUTED,
                                font=("TkDefaultFont", 11, "bold"))
        self.canvas.create_oval(x - 23, y - 23, x + 23, y + 23, outline="#34524b", width=2)
        if progress > 0:
            self.canvas.create_arc(x - 23, y - 23, x + 23, y + 23, start=90,
                                   extent=-min(progress * 360, 359.9), outline=color, width=3, style="arc")
        self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill=color, outline="")
        instruction = "Regardez le centre et relâchez les sourcils" if neutral else (
            f"Fixez le point  ·  {self.target_index + 1}/{len(TARGETS)}"
        )
        if not tracking:
            instruction = "Gardez les deux yeux visibles"
        self.canvas.create_text(width / 2, height - 43, text=instruction, fill=TEXT,
                                font=("TkDefaultFont", 15))
        self.canvas.create_text(width / 2, height - 19, text="Échap pour annuler", fill=MUTED)

    def sample_target(self, now: float, face: Observation | None, fresh: bool) -> None:
        if self.target_index is None:
            return
        neutral = self.target_index == -1
        samples = self.neutral_samples if neutral else self.samples[-1][1]
        if face is None or face.rays is None:
            if self.last_usable is not None and now - self.last_usable > 0.45:
                samples.clear()
                self.target_since = now
                self.last_usable = None
            self.draw_target(len(samples) / 14, False)
            return
        if fresh:
            self.last_usable = now
        if fresh and now - self.target_since >= 0.65:
            samples.append(face.brows if neutral else face.rays)
        self.draw_target(len(samples) / 14)
        if len(samples) < 14:
            return

        self.target_index += 1
        self.target_since = now
        if self.target_index < len(TARGETS):
            self.samples.append((TARGETS[self.target_index], []))
            return

        try:
            aspect = self.canvas.winfo_width() / self.canvas.winfo_height()
            calibration = calibrate(self.samples, aspect)
        except (ValueError, np.linalg.LinAlgError) as exc:
            self.end_calibration()
            messagebox.showwarning("Calibration incomplète", str(exc))
            return

        self.calibration = calibration
        self.calibrated_size = (self.canvas.winfo_width(), self.canvas.winfo_height())
        self.brow_baseline = np.median(self.neutral_samples, axis=0)
        if calibration.plane is None:
            label = "Pointeur calibré · plan 3D non déterminé"
        else:
            plane = calibration.plane
            dimensions = f"{np.linalg.norm(plane.right):.0f} × {np.linalg.norm(plane.down):.0f}"
            label = f"Écran 3D estimé · {dimensions} mm\nCentre {self.format_position(plane.center)}"
        self.result.set(label)
        self.notice = "Calibration terminée. Activez le pointeur pour le contrôler avec le regard."
        self.mouse_button.configure(state="normal")
        self.end_calibration()

    def end_calibration(self) -> None:
        self.target_index = None
        self.cal_window.destroy()

    def pause_mouse(self) -> None:
        self.mouse = None
        self.filtered = None
        self.clicks.reset()
        self.mouse_button.configure(text="Activer le pointeur")

    def toggle_mouse(self) -> None:
        if self.mouse is not None:
            self.pause_mouse()
            self.notice = "Pointeur en pause."
            return
        if self.calibration is None:
            return
        try:
            import pyautogui

            if sys.platform == "linux" and (
                self.root.tk.call("tk", "windowingsystem") != "x11"
                or os.environ.get("XDG_SESSION_TYPE") == "wayland"
            ):
                raise RuntimeError("Le contrôle du pointeur sous Linux nécessite une session X11.")
            self.screen_size = tuple(pyautogui.size())
            if self.screen_size != self.calibrated_size:
                raise RuntimeError("L’écran calibré doit être l’écran principal.")
            pyautogui.PAUSE = 0
        except Exception as exc:
            messagebox.showerror("Pointeur indisponible", f"Vérifiez les autorisations de contrôle de la souris.\n\n{exc}")
            return
        self.mouse = pyautogui
        self.mouse_button.configure(text="Mettre le pointeur en pause")
        self.notice = "Pointeur actif. Pour arrêter, placez la souris dans un coin de l’écran."

    def update_mouse(self, frame: Frame) -> None:
        if self.mouse is None or self.calibration is None or self.brow_baseline is None:
            return
        if frame.face is None or frame.face.rays is None:
            return
        mapped = self.calibration.predict(frame.face.rays)
        if mapped is None or np.any((mapped < -0.25) | (mapped > 1.25)):
            self.clicks.reset()
            self.filtered = None
            return
        # Aucune position générée par le regard ne peut déclencher le coin d'arrêt.
        x, y = np.clip(mapped, 0, 1) * (np.array(self.screen_size) - 3) + 1
        point = np.array([x, y])
        self.filtered = point if self.filtered is None else 0.28 * point + 0.72 * self.filtered
        lift = float(np.min(frame.face.brows - self.brow_baseline))
        click = self.clicks.update(frame.timestamp, lift, self.threshold.get())
        try:
            self.mouse.moveTo(*np.rint(self.filtered).astype(int).tolist())
            if click is not None:
                self.mouse.click(button=click)
        except self.mouse.FailSafeException:
            self.pause_mouse()
            self.notice = "Pointeur arrêté par le coin de sécurité."
        except Exception as exc:
            self.pause_mouse()
            self.notice = f"Pointeur arrêté. {exc}"

    @staticmethod
    def format_position(point: np.ndarray) -> str:
        return "({:.0f}, {:.0f}, {:.0f})".format(*point)

    def center_preview(self, event) -> None:
        for item in (self.preview_image, self.preview_label):
            self.preview.coords(item, event.width / 2, event.height / 2)

    def update_preview(self, frame: Frame) -> None:
        width = max(160, self.preview.winfo_width())
        height = max(120, self.preview.winfo_height())
        raw = Image.fromarray(cv2.cvtColor(cv2.flip(frame.image, 1), cv2.COLOR_BGR2RGB))
        # L'image n'impose pas de taille minimale à la grille lors d'un redimensionnement.
        raw.thumbnail((max(1, width - 4), max(1, height - 4)))
        self.photo = ImageTk.PhotoImage(raw)
        self.preview.itemconfigure(self.preview_image, image=self.photo)
        self.preview.itemconfigure(self.preview_label, state="hidden")
        if frame.face is not None:
            face = frame.face
            convergence = closest_approach(*face.rays) if face.rays is not None else None
            details = [f"Positions caméra (mm)",
                       f"Visage  {self.format_position(face.face)}",
                       f"Œil 1   {self.format_position(face.eyes[0])}",
                       f"Œil 2   {self.format_position(face.eyes[1])}"]
            if convergence is not None:
                details.append(f"Écart des rayons : {convergence.gap:.1f} mm")
            self.detail.set("\n".join(details))
        else:
            self.detail.set("Visage hors suivi")

    def tick(self) -> None:
        now = monotonic()
        frame = self.camera.latest
        current = frame is not None and now - frame.timestamp <= 0.35
        fresh = current and frame is not self.last_frame
        tracking = current and frame.face is not None and frame.face.rays is not None
        if self.camera.error:
            if self.target_index is not None:
                self.end_calibration()
            self.pause_mouse()
            self.status.set(self.camera.error)
            self.detail.set("En attente du visage")
            self.preview.itemconfigure(self.preview_image, image="")
            self.preview.itemconfigure(self.preview_label, state="normal")
            if not self.camera_button.winfo_manager():
                self.camera_button.pack(fill="x", pady=(5, 0), after=self.mouse_button)
        else:
            if fresh and tracking:
                self.last_gaze = frame.timestamp
            elif not current or frame.face is None or now - self.last_gaze > 0.20:
                self.clicks.reset()
                self.filtered = None
            self.status.set(self.notice if tracking else (
                "Montrez vos deux yeux à la caméra." if frame is not None else "Ouverture de la caméra…"
            ))
            if self.target_index is not None:
                self.sample_target(now, frame.face if current else None, fresh)
            elif fresh:
                self.update_mouse(frame)

            if fresh and self.target_index is None and now - self.last_preview >= 0.083:
                self.last_preview = now
                self.update_preview(frame)
        self.last_frame = frame
        self.root.after(25, self.tick)

    def escape(self, _event=None) -> None:
        if self.target_index is not None:
            self.end_calibration()
        else:
            self.pause_mouse()
            self.notice = "Pointeur en pause."

    def close(self) -> None:
        self.pause_mouse()
        self.camera.stop.set()
        self.camera.join(timeout=1)
        self.root.destroy()


def main() -> int:
    parser = argparse.ArgumentParser(description="Suivi du regard et des sourcils par webcam.")
    parser.add_argument("--camera", type=int, default=0, help="indice OpenCV de la webcam (0 par défaut)")
    parser.add_argument("--fov", type=float, default=65, help="champ de vision horizontal de la webcam en degrés")
    args = parser.parse_args()
    if args.camera < 0 or not 25 <= args.fov <= 120:
        parser.error("la caméra doit être positive ou nulle et le champ de vision entre 25° et 120°")

    if sys.platform == "win32":
        import ctypes

        ctypes.windll.user32.SetProcessDPIAware()
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(f"Affichage graphique indisponible. Lancez Eye Tracker depuis une session de bureau. ({exc})",
              file=sys.stderr)
        return 1
    Application(root, args.camera, args.fov)
    root.mainloop()
    return 0
