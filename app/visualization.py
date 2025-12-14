# app/visualization.py
from PIL import Image, ImageDraw
import numpy as np

# Соединения суставов (упрощённо)
SKELETON = [
    (11, 13), (13, 15),  # левая рука
    (12, 14), (14, 16),  # правая рука
    (11, 12),            # плечи
    (11, 23), (12, 24),  # корпус
    (23, 25), (25, 27),  # левая нога
    (24, 26), (26, 28),  # правая нога
    (23, 24)
]

def draw_skeleton_pil(frame, landmarks):
    """
    frame: numpy array (H, W, 3)
    landmarks: список [x, y] в пикселях
    """
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    pts = []
    for lm in landmarks:
        x, y = int(lm[0]), int(lm[1])
        pts.append((x, y))
        draw.ellipse((x-3, y-3, x+3, y+3), fill="red")

    for a, b in SKELETON:
        if a < len(pts) and b < len(pts):
            draw.line((pts[a], pts[b]), fill="green", width=2)

    return img

def make_side_by_side_pil(img1, img2):
    w1, h1 = img1.size
    w2, h2 = img2.size
    out = Image.new("RGB", (w1 + w2, max(h1, h2)), (255, 255, 255))
    out.paste(img1, (0, 0))
    out.paste(img2, (w1, 0))
    return out
