import cv2
import numpy as np
from pathlib import Path

Path("images").mkdir(exist_ok=True)

# Білий фон
image = np.full((650, 950, 3), 255, dtype=np.uint8)

# Трикутник
triangle = np.array([
    [80, 220],
    [180, 60],
    [280, 220]
])
cv2.fillPoly(image, [triangle], (0, 0, 0))

# Квадрат
cv2.rectangle(image, (370, 70), (530, 230), (0, 0, 0), -1)

# Прямокутник
cv2.rectangle(image, (630, 100), (880, 210), (0, 0, 0), -1)

# П'ятикутник
pentagon = np.array([
    [170, 330],
    [280, 410],
    [240, 540],
    [100, 540],
    [60, 410]
])
cv2.fillPoly(image, [pentagon], (0, 0, 0))

# Коло
cv2.circle(image, (650, 430), 105, (0, 0, 0), -1)

cv2.imwrite("images/test_shapes.png", image)

print("Створено images/test_shapes.png")