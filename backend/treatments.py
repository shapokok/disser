"""Treatment / care recommendations per class (ru + en) from treatment_database.json."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from labels import describe_class

_FILE = Path(__file__).resolve().parent / "treatment_database.json"

_GENERIC = {
    "en": {
        "severity": "unknown",
        "symptoms": "No specific description is available for this class.",
        "treatments": [
            "Consult a local agricultural extension service or plant pathologist",
            "Isolate affected plants and remove heavily damaged tissue",
            "Keep the canopy dry and improve air circulation",
        ],
        "prevention": ["Rotate crops", "Use certified disease-free planting material", "Inspect plants weekly"],
        "organic_options": ["Copper- or sulfur-based products approved for organic farming"],
    },
    "ru": {
        "severity": "неизвестно",
        "symptoms": "Для этого класса нет отдельного описания.",
        "treatments": [
            "Обратитесь в агрономическую службу или к фитопатологу",
            "Изолируйте поражённые растения и удалите сильно повреждённые ткани",
            "Обеспечьте проветривание и сухость листовой массы",
        ],
        "prevention": ["Соблюдайте севооборот", "Используйте здоровый посадочный материал", "Осматривайте растения еженедельно"],
        "organic_options": ["Медь- или серосодержащие препараты, разрешённые в органическом земледелии"],
    },
}


@lru_cache(maxsize=1)
def database() -> dict:
    with open(_FILE, encoding="utf-8") as f:
        return json.load(f)


def get_treatment(class_name: str, lang: str = "en") -> dict:
    lang = "ru" if lang == "ru" else "en"
    info = describe_class(class_name)
    entry = database().get(class_name, {}).get(lang)
    if entry is None:
        entry = dict(_GENERIC[lang])
    return {
        "class_raw": class_name,
        "disease_name": info["disease_ru"] if lang == "ru" else info["disease"],
        "plant": info["plant_ru"] if lang == "ru" else info["plant"],
        "healthy": info["healthy"],
        "lang": lang,
        "generic": class_name not in database(),
        **entry,
    }
