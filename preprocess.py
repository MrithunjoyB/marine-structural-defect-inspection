"""Image preprocessing utilities for structural inspection images."""

from __future__ import annotations

import cv2
import numpy as np


def resize_image(image: np.ndarray, width: int = 1024) -> np.ndarray:
    height, current_width = image.shape[:2]
    if current_width <= width:
        return image.copy()
    scale = width / current_width
    new_size = (width, int(height * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def denoise_image(image: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(image, None, h=8, hColor=8, templateWindowSize=7, searchWindowSize=21)


def sharpen_image(image: np.ndarray) -> np.ndarray:
    kernel = np.array([[0, -1, 0], [-1, 5.2, -1], [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(image, -1, kernel)


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def grayscale_preview(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def edge_detection_preview(image: np.ndarray) -> np.ndarray:
    gray = grayscale_preview(image)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blurred, 70, 170)


def adjust_brightness_contrast(image: np.ndarray, brightness: int = 0, contrast: int = 0) -> np.ndarray:
    """Adjust brightness and contrast using OpenCV's alpha/beta transform."""

    alpha = 1.0 + (contrast / 100.0)
    beta = brightness
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def apply_preprocessing(
    image: np.ndarray,
    resize_width: int = 1024,
    denoise: bool = True,
    clahe: bool = True,
    sharpen: bool = False,
    brightness: int = 0,
    contrast: int = 0,
) -> np.ndarray:
    processed = resize_image(image, resize_width)
    if brightness or contrast:
        processed = adjust_brightness_contrast(processed, brightness, contrast)
    if denoise:
        processed = denoise_image(processed)
    if clahe:
        processed = enhance_contrast(processed)
    if sharpen:
        processed = sharpen_image(processed)
    return processed


def build_preview_grid(image: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "Processed": image,
        "Grayscale": grayscale_preview(image),
        "Edges": edge_detection_preview(image),
        "CLAHE Enhanced": enhance_contrast(image),
    }
