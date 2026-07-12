"""Deterministic expanded synthetic benchmark generation and validation."""

from __future__ import annotations

import hashlib

import cv2
import numpy as np


EXPANDED_CATEGORIES = (
    "thin_crack",
    "pitting_cluster",
    "weld_disturbance",
    "colour_only_anomaly",
    "texture_only_anomaly",
    "normal_texture",
    "specular_highlights",
    "illumination_gradient",
    "border_artifact",
    "blur_noise",
)
POSITIVE_CATEGORIES = frozenset(EXPANDED_CATEGORIES[:5])


def derived_seed(master_seed: int, category: str, sample_index: int) -> int:
    payload = f"expanded-v1:{master_seed}:{category}:{sample_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def generate_expanded_cases(
    master_seed: int = 42,
    samples_per_category: int = 50,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, dict[str, object]]]:
    """Generate category-balanced images independently of proposal outputs."""
    cases: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    parameters: dict[str, dict[str, object]] = {}
    for category in EXPANDED_CATEGORIES:
        for index in range(samples_per_category):
            seed = derived_seed(master_seed, category, index)
            rng = np.random.default_rng(seed)
            image, mask, details = _generate_case(category, rng, index)
            name = f"{category}_{index + 1:03d}"
            template_index = index // 5
            cases[name] = (image, mask)
            parameters[name] = {
                "master_seed": master_seed,
                "derived_seed": seed,
                "sample_index": index,
                "category": category,
                "anomaly_type": category,
                "image_outcome": "anomaly_present" if category in POSITIVE_CATEGORIES else "no_anomaly",
                "source_group_id": f"expanded-source:{category}:{template_index:02d}",
                "template_group_id": f"expanded-template:{category}:{template_index:02d}",
                "near_duplicate_group_id": "",
                "provenance": "Deterministic local generator; expanded synthetic benchmark v1",
                "licence": "MIT-compatible generated data",
                **details,
            }
    hashes = [_png_hash(image) for image, _ in cases.values()]
    if len(hashes) != len(set(hashes)):
        raise AssertionError("Expanded synthetic generation produced an exact duplicate")
    return cases, parameters


def validate_generated_cases(
    cases: dict[str, tuple[np.ndarray, np.ndarray]],
    parameters: dict[str, dict[str, object]],
    expected_per_category: int,
) -> dict[str, object]:
    categories = {category: 0 for category in EXPANDED_CATEGORIES}
    hashes: list[str] = []
    phashes: list[str] = []
    failures: list[str] = []
    for name, (image, mask) in cases.items():
        details = parameters.get(name, {})
        category = str(details.get("category", ""))
        if category not in categories:
            failures.append(f"{name}: invalid category")
            continue
        categories[category] += 1
        if image.shape != (300, 500, 3) or image.dtype != np.uint8 or float(image.std()) < 2:
            failures.append(f"{name}: blank or invalid image")
        if mask.shape != image.shape[:2] or mask.dtype != np.uint8:
            failures.append(f"{name}: invalid mask dimensions")
        has_mask = bool(np.any(mask))
        if category in POSITIVE_CATEGORIES and not has_mask:
            failures.append(f"{name}: positive mask is empty")
        if category not in POSITIVE_CATEGORIES and has_mask:
            failures.append(f"{name}: clean mask is non-empty")
        required = {"derived_seed", "source_group_id", "template_group_id", "image_outcome", "provenance", "licence"}
        if not required.issubset(details):
            failures.append(f"{name}: incomplete metadata")
        hashes.append(_png_hash(image))
        phashes.append(_perceptual_hash(image))
    if any(count != expected_per_category for count in categories.values()):
        failures.append("category balance mismatch")
    if len(hashes) != len(set(hashes)):
        failures.append("exact duplicate image hash")
    return {
        "passed": not failures,
        "image_count": len(cases),
        "category_counts": categories,
        "unique_sha256_count": len(set(hashes)),
        "unique_perceptual_hash_count": len(set(phashes)),
        "failures": failures,
    }


def _generate_case(category: str, rng: np.random.Generator, index: int):
    h, w = 300, 500
    level = float(rng.uniform(105, 195))
    texture_sigma = float(rng.uniform(3, 18))
    illumination = float(rng.uniform(-35, 35))
    vertical = float(rng.uniform(-18, 18))
    noise = rng.normal(0, texture_sigma, (h, w, 1))
    x_ramp = np.linspace(-illumination, illumination, w)[None, :, None]
    y_ramp = np.linspace(-vertical, vertical, h)[:, None, None]
    tint = rng.uniform(-12, 12, 3).reshape(1, 1, 3)
    image = np.clip(level + np.repeat(noise + x_ramp + y_ramp, 3, axis=2) + tint, 0, 255).astype(np.uint8)
    mask = np.zeros((h, w), np.uint8)
    cx, cy = int(rng.integers(70, w - 70)), int(rng.integers(55, h - 55))
    angle = float(rng.uniform(-88, 88))
    contrast = float(rng.uniform(12, 125))
    bright = bool(rng.integers(0, 2))
    value = int(np.clip(level + contrast if bright else level - contrast, 8, 250))
    blur_sigma = float(rng.uniform(0, 1.3))
    details: dict[str, object] = {
        "position": [cx, cy], "orientation": angle, "contrast": contrast,
        "background_level": level, "background_texture_sigma": texture_sigma,
        "illumination": illumination, "exposure": float(level / 255),
        "base_colour_offset": tint.flatten().round(3).tolist(), "noise_sigma": texture_sigma,
        "blur_sigma": blur_sigma, "difficulty_band": ["low", "medium", "high"][index % 3],
    }

    if category == "thin_crack":
        length, width = int(rng.integers(90, 390)), int(rng.integers(1, 7))
        points = _crack_points(rng, (cx, cy), length, angle, bool(index % 3 == 0), w, h)
        cv2.polylines(image, [points], False, (value,) * 3, width, cv2.LINE_AA)
        cv2.polylines(mask, [points], False, 255, width + 3, cv2.LINE_AA)
        if index % 4 == 0:
            branch = points[len(points) // 2]
            end = (int(np.clip(branch[0] + rng.integers(-65, 66), 0, w - 1)), int(np.clip(branch[1] + rng.integers(-55, 56), 0, h - 1)))
            cv2.line(image, tuple(branch), end, (value,) * 3, max(1, width - 1), cv2.LINE_AA)
            cv2.line(mask, tuple(branch), end, 255, width + 2, cv2.LINE_AA)
        details.update({"anomaly_size": length, "crack_width": width, "bright_or_reflective": bright, "curved": len(points) > 2, "branched": index % 4 == 0})
    elif category == "pitting_cluster":
        pit_count = int(rng.integers(3, 16)); spread = int(rng.integers(25, 90)); reflective = bool(index % 4 == 0)
        radii = []
        for _ in range(pit_count):
            point = (int(np.clip(cx + rng.normal(0, spread / 2), 8, w - 9)), int(np.clip(cy + rng.normal(0, spread / 2), 8, h - 9)))
            radius = int(rng.integers(3, 15)); radii.append(radius)
            pit_value = int(np.clip(value + (rng.uniform(45, 100) if reflective else rng.uniform(-15, 15)), 5, 252))
            cv2.circle(image, point, radius, (pit_value,) * 3, -1, cv2.LINE_AA)
            cv2.circle(mask, point, radius + 2, 255, -1, cv2.LINE_AA)
        details.update({"anomaly_size": radii, "pit_count": pit_count, "cluster_spread": spread, "reflective": reflective})
    elif category == "weld_disturbance":
        axes = (int(rng.integers(55, 175)), int(rng.integers(10, 38)))
        colour = tuple(int(np.clip(value + offset, 0, 255)) for offset in rng.uniform(-20, 25, 3))
        cv2.ellipse(image, (cx, cy), axes, angle, 0, 360, colour, -1, cv2.LINE_AA)
        cv2.ellipse(mask, (cx, cy), axes, angle, 0, 360, 255, -1, cv2.LINE_AA)
        details.update({"anomaly_size": list(axes), "structural_complexity": float(axes[0] / axes[1])})
    elif category == "colour_only_anomaly":
        axes = (int(rng.integers(20, 85)), int(rng.integers(12, 55)))
        hsv = np.uint8([[[int(rng.integers(0, 180)), int(rng.integers(45, 210)), int(np.clip(level + rng.uniform(-35, 35), 30, 235))]]])
        colour = tuple(int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])
        cv2.ellipse(image, (cx, cy), axes, angle, 0, 360, colour, -1, cv2.LINE_AA)
        cv2.ellipse(mask, (cx, cy), axes, angle, 0, 360, 255, -1, cv2.LINE_AA)
        details.update({"anomaly_size": list(axes), "anomaly_colour_bgr": colour})
    elif category == "texture_only_anomaly":
        axes = (int(rng.integers(22, 90)), int(rng.integers(15, 60)))
        local = np.zeros_like(mask); cv2.ellipse(local, (cx, cy), axes, angle, 0, 360, 255, -1)
        texture = rng.normal(level, rng.uniform(25, 70), image.shape).astype(np.float32)
        image[local > 0] = np.clip(texture[local > 0], 0, 255).astype(np.uint8); mask = local
        details.update({"anomaly_size": list(axes), "local_texture_sigma": float(np.std(texture[local > 0]))})
    elif category == "specular_highlights":
        count = int(rng.integers(1, 7)); elongated = bool(index % 2); sharp = bool(index % 3)
        exposure = float(rng.uniform(0.68, 1.0)); sizes = []
        layer = image.copy()
        for _ in range(count):
            point = (int(rng.integers(25, w - 25)), int(rng.integers(20, h - 20)))
            axes = (int(rng.integers(5, 40)), int(rng.integers(3, 15)) if elongated else int(rng.integers(5, 35)))
            highlight = int(np.clip(175 + 90 * exposure + rng.uniform(-20, 15), 170, 255)); sizes.append(list(axes))
            cv2.ellipse(layer, point, axes, float(rng.uniform(-90, 90)), 0, 360, (highlight,) * 3, -1, cv2.LINE_AA)
        if not sharp: layer = cv2.GaussianBlur(layer, (0, 0), float(rng.uniform(1.2, 4.5)))
        image = layer
        details.update({"highlight_count": count, "highlight_sizes": sizes, "elongated": elongated, "sharp_boundary": sharp, "highlight_exposure": exposure, "partial_saturation": exposure < .9})
    elif category == "illumination_gradient":
        strength = float(rng.uniform(35, 105)); direction = float(rng.uniform(-np.pi, np.pi))
        xx, yy = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h)); field = (np.cos(direction) * xx + np.sin(direction) * yy) * strength
        image = np.clip(image.astype(np.float32) + field[..., None], 0, 255).astype(np.uint8)
        details.update({"gradient_strength": strength, "gradient_direction": direction})
    elif category == "border_artifact":
        side = ["top", "bottom", "left", "right"][index % 4]; width = int(rng.integers(5, 38)); border_value = int(rng.integers(0, 35))
        if side == "top": image[:width] = border_value
        elif side == "bottom": image[-width:] = border_value
        elif side == "left": image[:, :width] = border_value
        else: image[:, -width:] = border_value
        details.update({"border_side": side, "border_width": width, "border_value": border_value})
    elif category == "blur_noise":
        sigma = float(rng.uniform(0.8, 5.0)); noise_sigma = float(rng.uniform(4, 35))
        if index % 2: image = cv2.GaussianBlur(image, (0, 0), sigma)
        image = np.clip(image.astype(np.float32) + rng.normal(0, noise_sigma, image.shape), 0, 255).astype(np.uint8)
        details.update({"applied_blur_sigma": sigma if index % 2 else 0.0, "applied_noise_sigma": noise_sigma})
    elif category == "normal_texture":
        details.update({"normal_texture_family": index % 5})

    if blur_sigma > .15 and category not in {"specular_highlights", "blur_noise"}:
        image = cv2.GaussianBlur(image, (0, 0), blur_sigma)
    mask = ((mask > 0).astype(np.uint8) * 255)
    return image, mask, details


def _crack_points(rng, centre, length, angle, curved, width, height):
    direction = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
    normal = np.array([-direction[1], direction[0]])
    samples = 7 if curved else 2
    points = []
    for fraction in np.linspace(-.5, .5, samples):
        bend = np.sin((fraction + .5) * np.pi) * rng.uniform(-length * .12, length * .12) if curved else 0
        point = np.array(centre) + direction * length * fraction + normal * bend
        points.append((int(np.clip(point[0], 1, width - 2)), int(np.clip(point[1], 1, height - 2))))
    return np.asarray(points, np.int32)


def _png_hash(image: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Could not encode generated image")
    return hashlib.sha256(encoded.tobytes()).hexdigest()


def _perceptual_hash(image: np.ndarray) -> str:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (9, 8))
    return "".join("1" if bit else "0" for bit in (resized[:, 1:] > resized[:, :-1]).flat)
