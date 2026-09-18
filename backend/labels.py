"""
Human readable class labels (English / Russian) for the 38 PlantVillage classes.
The source of truth is class_labels.json next to this file.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

_FILE = Path(__file__).resolve().parent / "class_labels.json"


@lru_cache(maxsize=1)
def all_labels() -> dict:
    with open(_FILE, encoding="utf-8") as f:
        return json.load(f)


def _fallback(class_name: str) -> dict:
    plant, _, disease = class_name.partition("___")
    plant = plant.replace("_", " ").replace(",", ",")
    disease = disease.replace("_", " ").strip() or "unknown"
    return {"plant_en": plant, "disease_en": disease, "plant_ru": plant, "disease_ru": disease}


def label_info(class_name: str) -> dict:
    return all_labels().get(class_name) or _fallback(class_name)


def format_class(class_name: str, lang: str = "en") -> str:
    """'Tomato___Late_blight' -> 'Tomato — Late blight' / 'Томат — Фитофтороз'."""
    info = label_info(class_name)
    suffix = "ru" if lang == "ru" else "en"
    return f"{info['plant_' + suffix]} — {info['disease_' + suffix]}"


def describe_class(class_name: str) -> dict:
    """All label variants for API responses."""
    info = label_info(class_name)
    return {
        "class_raw": class_name,
        "class": format_class(class_name, "en"),
        "class_ru": format_class(class_name, "ru"),
        "plant": info["plant_en"],
        "plant_ru": info["plant_ru"],
        "disease": info["disease_en"],
        "disease_ru": info["disease_ru"],
        "healthy": class_name.lower().endswith("healthy"),
    }
