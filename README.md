# Eye Tracker

Eye Tracker utilise une webcam pour estimer la position du visage et des yeux en 3D et déplacer le pointeur de l’écran principal. Une levée des deux sourcils déclenche un clic gauche. Deux levées rapprochées, avec un retour au repos entre les deux, déclenchent un clic droit.

## Installation

Sur l’ordinateur auquel la webcam et l’écran sont connectés, installez **Python 3.12** avec Tkinter. Ouvrez un terminal dans le dossier du projet.

**Windows (PowerShell)**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m eyetracker
```

**macOS ou Linux (X11)**

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m eyetracker
```

Sous macOS, autorisez l’accès à la caméra et, pour contrôler la souris, accordez l’autorisation **Accessibilité** au terminal ou à l’application qui lance Python. Sous Linux, le contrôle de la souris nécessite une session X11 et un Python compilé avec Tkinter. MediaPipe fournit des paquets pour les plateformes listées sur PyPI. [Paquets MediaPipe](https://pypi.org/project/mediapipe/0.10.21/#files)

L’option `--camera 1` sélectionne la deuxième webcam si nécessaire. L’option `--fov 70` ajuste le champ de vision **horizontal** si la valeur de votre webcam est connue. Le défaut est de 65°.

## Calibration et clics

1. Placez la webcam près de l’écran principal. Asseyez-vous en face, à environ 50–75 cm, avec un éclairage uniforme sur le visage.
2. Cliquez sur **Calibrer l’écran**. Regardez le premier point au centre en gardant les sourcils au repos, puis fixez chaque repère jusqu’au suivant. Gardez la tête aussi stable que possible.
3. Cliquez sur **Activer le pointeur**. Regardez une zone de l’écran, stabilisez le regard, puis levez les deux sourcils. Le clic gauche attend environ 650 ms pour laisser le temps d’un deuxième geste. Pour le clic droit, relâchez les sourcils puis relevez-les dans ce délai. Si les gestes sont difficiles à déclencher, ajustez le **Seuil de levée des sourcils**.
4. Pour mettre en pause, déplacez la souris physique dans un coin de l’écran ou pressez **Échap** lorsque la fenêtre a le focus. Recalibrez après avoir déplacé la caméra ou l’écran.

## Vue 3D

Cliquez sur **Ouvrir la vue 3D** pour voir le visage, les axes de la webcam et les deux rayons du regard en direct. Après une calibration 3D réussie, l’écran apparaît comme un rectangle bleu. Les points colorés sur ce rectangle indiquent où les rayons rencontrent sa surface. Le point jaune indique le milieu du plus court segment entre les deux rayons lorsque leur convergence est exploitable.

Glissez dans la vue pour tourner autour de la scène, utilisez la molette pour zoomer et cliquez sur **Recentrer** pour revenir au cadrage d’origine. L’ouverture de la vue met le pointeur en pause.

## Géométrie

MediaPipe détecte les iris, les paupières et les sourcils. OpenCV ajuste un modèle du visage aux repères vus par la caméra. Un globe oculaire approximatif sert à convertir chaque iris en rayon 3D. Les coordonnées affichées sont dans le repère de la caméra, avec X vers la droite de l’image brute, Y vers le bas et Z vers l’avant de la caméra.

Pour chaque repère de calibration, l’application mesure le plus court segment entre les deux demi-droites du regard et retient son milieu. Un rectangle au format de l’écran est ajusté aux milieux utilisables. Les dimensions et la profondeur sont des estimations fondées sur une webcam monoculaire, un visage de référence et un champ de vision approximatif. Si les convergences sont trop dispersées, l’application l’indique et peut conserver un calibrage 2D du pointeur.

Les images et la calibration sont traitées localement. La calibration est conservée pendant la session.

Les repères 3D de référence proviennent du modèle canonique de MediaPipe, distribué sous licence Apache 2.0. [Modèle canonique](https://github.com/google-ai-edge/mediapipe/blob/v0.10.21/mediapipe/modules/face_geometry/data/canonical_face_model.obj)

## Tests

Depuis l’environnement Python installé, exécutez `python -m unittest discover -s tests -v` pour vérifier la géométrie, la calibration et les deux gestes sans webcam.
