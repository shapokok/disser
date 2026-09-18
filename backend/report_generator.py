"""
PDF reports (ReportLab) with Cyrillic support.

DejaVu Sans ships with matplotlib, so no extra font download is required.
"""

from __future__ import annotations

import base64
import io
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image as RLImage
from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

_FONT = "Helvetica"
_FONT_BOLD = "Helvetica-Bold"


def _register_fonts():
    global _FONT, _FONT_BOLD
    if _FONT == "DejaVuSans":
        return
    try:
        import matplotlib

        ttf = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
        pdfmetrics.registerFont(TTFont("DejaVuSans", str(ttf / "DejaVuSans.ttf")))
        pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", str(ttf / "DejaVuSans-Bold.ttf")))
        _FONT, _FONT_BOLD = "DejaVuSans", "DejaVuSans-Bold"
    except Exception as e:  # pragma: no cover
        print(f"Warning: Cyrillic PDF font unavailable ({e})")


T = {
    "en": {
        "title": "Crop Disease Detection Report",
        "generated": "Generated",
        "image": "Image",
        "prediction": "Prediction",
        "confidence": "Confidence",
        "model": "Model",
        "explanation": "Explanation",
        "mode": "Mode",
        "time": "Inference time",
        "top": "Alternative classes",
        "treatment": "Recommendations",
        "symptoms": "Symptoms",
        "treatments": "Treatment",
        "prevention": "Prevention",
        "organic": "Organic options",
        "severity": "Severity",
        "comparison": "Model comparison",
        "class": "Class",
        "agreement": "Models agreeing",
        "original": "Original",
        "overlay": "Grad-CAM overlay",
        "lime": "LIME regions",
        "footer": "Master's thesis: intelligent crop condition monitoring with computer vision and neural networks",
    },
    "ru": {
        "title": "Отчёт о диагностике заболеваний растений",
        "generated": "Сформирован",
        "image": "Изображение",
        "prediction": "Диагноз",
        "confidence": "Уверенность",
        "model": "Модель",
        "explanation": "Метод объяснения",
        "mode": "Режим",
        "time": "Время инференса",
        "top": "Альтернативные классы",
        "treatment": "Рекомендации",
        "symptoms": "Симптомы",
        "treatments": "Лечение",
        "prevention": "Профилактика",
        "organic": "Органические меры",
        "severity": "Серьёзность",
        "comparison": "Сравнение моделей",
        "class": "Класс",
        "agreement": "Согласие моделей",
        "original": "Оригинал",
        "overlay": "Наложение Grad-CAM",
        "lime": "Области LIME",
        "footer": "Магистерская диссертация: интеллектуальная система мониторинга состояния сельскохозяйственных культур",
    },
}


def _styles():
    _register_fonts()
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=base["Title"], fontName=_FONT_BOLD, fontSize=20, textColor=colors.HexColor("#1f5130"), alignment=TA_CENTER, spaceAfter=4),
        "meta": ParagraphStyle("m", parent=base["Normal"], fontName=_FONT, fontSize=9, textColor=colors.grey, alignment=TA_CENTER, spaceAfter=10),
        "h2": ParagraphStyle("h", parent=base["Heading2"], fontName=_FONT_BOLD, fontSize=13, textColor=colors.HexColor("#1f2933"), spaceBefore=8, spaceAfter=6),
        "h3": ParagraphStyle("h3", parent=base["Heading3"], fontName=_FONT_BOLD, fontSize=10.5, spaceBefore=6, spaceAfter=3),
        "body": ParagraphStyle("b", parent=base["Normal"], fontName=_FONT, fontSize=9.5, leading=13),
        "small": ParagraphStyle("s", parent=base["Normal"], fontName=_FONT, fontSize=8, textColor=colors.grey),
    }


def _img(b64: str | None, width_mm: float):
    if not b64:
        return None
    try:
        from PIL import Image as PILImage

        data = base64.b64decode(b64)
        w, h = PILImage.open(io.BytesIO(data)).size
        return RLImage(io.BytesIO(data), width=width_mm * mm, height=width_mm * mm * h / w)
    except Exception:
        return None


def _table(rows, col_widths, header=True):
    t = Table(rows, colWidths=col_widths, hAlign="LEFT")
    style = [
        ("FONTNAME", (0, 0), (-1, -1), _FONT),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d0d7de")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if header:
        style += [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8f3ec")),
            ("FONTNAME", (0, 0), (-1, 0), _FONT_BOLD),
        ]
    t.setStyle(TableStyle(style))
    return t


def _cls(entry: dict, lang: str) -> str:
    return entry.get("class_ru" if lang == "ru" else "class") or entry.get("class", "")


def _bullets(items, st):
    return [Paragraph(f"• {x}", st["body"]) for x in (items or [])]


def _doc(buf, title):
    return SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm, title=title)


def create_pdf_report(results: list[dict], lang: str = "en") -> bytes:
    lang = "ru" if lang == "ru" else "en"
    t, st = T[lang], _styles()
    buf = io.BytesIO()
    doc = _doc(buf, t["title"])
    story = [Paragraph(t["title"], st["title"]), Paragraph(f"{t['generated']}: {datetime.now():%Y-%m-%d %H:%M}", st["meta"])]

    for n, r in enumerate(results, 1):
        pred = r.get("prediction", {})
        block = [Paragraph(f"{n}. {r.get('image_name', t['image'])}", st["h2"])]

        imgs = r.get("images") or {}
        pics = [(t["original"], imgs.get("original")), (t["overlay"], imgs.get("overlay")), (t["lime"], imgs.get("lime"))]
        pics = [(cap, b) for cap, b in pics if b]
        if pics:
            width = min(55.0, 170.0 / len(pics) - 4)
            cells = [_img(b, width) or "" for _, b in pics]
            caps = [Paragraph(cap, st["small"]) for cap, _ in pics]
            block.append(_table([cells, caps], [(width + 4) * mm] * len(pics), header=False))
            block.append(Spacer(1, 4))

        info = [
            [t["prediction"], _cls(pred, lang)],
            [t["confidence"], pred.get("confidence_percent", "")],
            [t["model"], r.get("model_label") or r.get("model", "")],
            [t["explanation"], (r.get("explanation") or "").upper()],
            [t["mode"], r.get("dataset_type", "")],
            [t["time"], f"{r.get('inference_time_ms', '')} ms"],
        ]
        block.append(_table([[Paragraph(f"<b>{a}</b>", st["body"]), Paragraph(str(b), st["body"])] for a, b in info], [45 * mm, 120 * mm], header=False))

        top = r.get("top_predictions") or []
        if len(top) > 1:
            block.append(Paragraph(t["top"], st["h3"]))
            block.append(_table([[t["class"], t["confidence"]]] + [[_cls(x, lang), x.get("confidence_percent", "")] for x in top[:5]], [120 * mm, 45 * mm]))

        tr = r.get("treatment")
        if tr:
            block.append(Paragraph(f"{t['treatment']}: {tr.get('disease_name', '')}", st["h3"]))
            if tr.get("severity"):
                block.append(Paragraph(f"<b>{t['severity']}:</b> {tr['severity']}", st["body"]))
            if tr.get("symptoms"):
                block.append(Paragraph(f"<b>{t['symptoms']}:</b> {tr['symptoms']}", st["body"]))
            for key, label in (("treatments", "treatments"), ("prevention", "prevention"), ("organic_options", "organic")):
                if tr.get(key):
                    block.append(Paragraph(f"<b>{t[label]}</b>", st["body"]))
                    block += _bullets(tr[key], st)
        block.append(Spacer(1, 8))
        story.append(KeepTogether(block[:3]))
        story += block[3:]

    story.append(Spacer(1, 10))
    story.append(Paragraph(t["footer"], st["small"]))
    doc.build(story)
    return buf.getvalue()


def create_comparison_report(data: dict, lang: str = "en") -> bytes:
    lang = "ru" if lang == "ru" else "en"
    t, st = T[lang], _styles()
    buf = io.BytesIO()
    doc = _doc(buf, t["comparison"])
    story = [Paragraph(t["comparison"], st["title"]), Paragraph(f"{t['generated']}: {datetime.now():%Y-%m-%d %H:%M}", st["meta"])]
    if data.get("image_name"):
        story.append(Paragraph(f"{t['image']}: {data['image_name']}", st["body"]))
    story.append(Spacer(1, 6))
    rows = [[t["model"], t["class"], t["confidence"], t["time"]]]
    for name, r in (data.get("comparisons") or {}).items():
        rows.append([r.get("label") or name, _cls(r, lang), r.get("confidence_percent", ""), f"{r.get('inference_time_ms', '')} ms"])
    story.append(_table(rows, [45 * mm, 75 * mm, 25 * mm, 25 * mm]))
    agr = data.get("agreement")
    if agr:
        story.append(Spacer(1, 6))
        story.append(Paragraph(f"{t['agreement']}: {agr.get('models_agreeing')}/{agr.get('models_total')} → {_cls(agr.get('class', {}), lang)}", st["body"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph(t["footer"], st["small"]))
    doc.build(story)
    return buf.getvalue()
