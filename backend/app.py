"""
Crop Disease Detection API + static frontend.

Run:   python backend/app.py              (or: flask --app backend.app run)
Env:   see backend/config.py (CROP_PORT, CROP_DEVICE, CROP_MODELS, ...)
"""

from __future__ import annotations

import io
import json
import sys
import time
import traceback
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
import export_utils
import validation
from explain import explain_gradcam, explain_lime, load_image, to_tensor
from field_model import FieldAdapter
from labels import describe_class
from model import MODEL_TYPES, MaskedModel, ModelManager
from report_generator import create_comparison_report, create_pdf_report
from treatments import get_treatment


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _lang() -> str:
    lang = request.args.get("lang") or (request.get_json(silent=True) or {}).get("lang") or "ru"
    return "ru" if str(lang).lower().startswith("ru") else "en"


def _error(message: str, status: int = 400, **extra):
    payload = {"success": False, "error": message, **extra}
    return jsonify(payload), status


def _percent(x: float) -> str:
    return f"{x * 100:.2f}%"


def create_app(load_models: bool = True, models=None) -> Flask:
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_FILE_SIZE
    app.config["JSON_AS_ASCII"] = False
    app.json.ensure_ascii = False
    CORS(app, resources={r"/api/*": {"origins": config.CORS_ORIGINS}})

    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "heatmaps").mkdir(parents=True, exist_ok=True)

    manager = ModelManager(config.MODELS_DIR, device=config.DEVICE, trained_threshold=config.TRAINED_ACCURACY_THRESHOLD)
    if load_models:
        for name in models or config.MODELS_TO_LOAD:
            try:
                manager.load_model(name, name)
            except Exception as e:  # keep serving with the models that did load
                print(f"Could not load model '{name}': {e}")
    field = FieldAdapter(manager.class_names, manager.device)
    app.extensions["manager"] = manager
    app.extensions["field"] = field
    print(f"Loaded models: {list(manager.models)}  field model: {'yes' if field.available else 'no'}")

    # ------------------------------------------------------------------ helpers
    def resolve_image(name: str) -> Path:
        if not name or not str(name).strip():
            raise ValueError("image_path is required")
        p = Path(name)
        if not p.is_absolute():
            p = config.UPLOAD_DIR / secure_filename(name)
        p = p.resolve()
        allowed_roots = (config.UPLOAD_DIR, config.DATA_DIR)
        if not any(str(p).startswith(str(root)) for root in allowed_roots):
            raise PermissionError("Image path is outside the allowed directories")
        if not p.is_file():
            raise FileNotFoundError(f"Image not found: {name}")
        return p

    def format_prediction(probs, idx: int, top_k: int = 5) -> tuple[dict, list]:
        order = probs.argsort()[::-1][:top_k]
        pred = {
            **describe_class(manager.class_names[idx]),
            "confidence": float(probs[idx]),
            "confidence_percent": _percent(float(probs[idx])),
        }
        top = [
            {
                **describe_class(manager.class_names[i]),
                "confidence": float(probs[i]),
                "confidence_percent": _percent(float(probs[i])),
            }
            for i in order
        ]
        return pred, top

    def run_prediction(
        image_path: Path, model_name: str, explanation: str, dataset_type: str, lang: str, ensemble_method=None
    ) -> dict:
        start = time.time()
        tensor = to_tensor(load_image(image_path))
        field_info = None
        ensemble_info = None

        if dataset_type == "field" and field.available:
            model = MaskedModel(field.model, field.covered_idx, len(manager.class_names)).to(manager.device).eval()
            target_layer = field.target_layer
            label = "MobileNet-V2 (domain-adapted)"
            field_info = {
                "adapted_model": True,
                "covered_classes": [describe_class(c) for c in field.covered_classes],
                "method": field.info.get("method"),
                "accuracy_field": field.info.get("accuracy_test"),
            }
        elif model_name == "ensemble":
            ens = manager.predict_ensemble(tensor, ensemble_method or "weighted")
            best = manager.best_model()
            model = manager.get_model(best)
            target_layer = manager.get_target_layer(best)
            label = "Ensemble"
            ensemble_info = {
                "method": ens["ensemble_method"],
                "models_used": ens["models_used"],
                "weights": ens["weights"],
                "explained_with": best,
                "agreement_rate": ens["agreement_rate"],
                "agreement_percent": f"{ens['agreement_rate'] * 100:.0f}%",
                "individual_predictions": {
                    n: {
                        **describe_class(p["predicted_class"]),
                        "confidence": p["confidence"],
                        "confidence_percent": _percent(p["confidence"]),
                    }
                    for n, p in ens["individual_predictions"].items()
                },
                "uncertainty_metrics": ens["uncertainty_metrics"],
            }
            if dataset_type == "field":
                field_info = {"adapted_model": False, "note": "field model unavailable"}
        else:
            if model_name not in manager.models:
                raise KeyError(f"Model '{model_name}' is not loaded")
            model = manager.get_model(model_name)
            target_layer = manager.get_target_layer(model_name)
            label = MODEL_TYPES[manager.models[model_name]["type"]]["label"]
            if dataset_type == "field":
                field_info = {"adapted_model": False, "note": "field model unavailable"}

        images = {}
        if explanation == "gradcam":
            out = explain_gradcam(model, target_layer, image_path, manager.device)
            probs, idx, images = out["probabilities"], out["predicted_idx"], out["images"]
        elif explanation == "lime":
            out = explain_lime(model, image_path, manager.device)
            probs, idx, images = out["probabilities"], out["predicted_idx"], out["images"]
        elif explanation == "none":
            import torch

            with torch.no_grad():
                import torch.nn.functional as F

                probs = F.softmax(model(tensor.to(manager.device)), dim=1)[0].float().cpu().numpy()
            idx = int(probs.argmax())
            from explain import encode_jpeg

            images = {"original": encode_jpeg(load_image(image_path))}
        else:
            raise ValueError(f"Unknown explanation method: {explanation}")

        if ensemble_info is not None:  # the ensemble decides, the best model explains
            probs, idx = ens["probabilities"], ens["predicted_idx"]

        prediction, top = format_prediction(probs, idx)
        return {
            "success": True,
            "image_name": image_path.name,
            "image_url": f"/api/uploads/{image_path.name}" if image_path.parent == config.UPLOAD_DIR else None,
            "model": model_name,
            "model_label": label,
            "explanation": explanation,
            "dataset_type": dataset_type,
            "prediction": prediction,
            "top_predictions": top,
            "images": images,
            "field_mode": field_info,
            "ensemble": ensemble_info,
            "treatment": get_treatment(prediction["class_raw"], lang),
            "inference_time_ms": round((time.time() - start) * 1000, 1),
            "timestamp": _now(),
        }

    @lru_cache(maxsize=1)
    def dataset_info() -> dict:
        counts = {}
        for split in ("train", "valid", "test"):
            d = config.PLANTVILLAGE_DIR / split
            counts[split] = (
                sum(1 for p in d.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}) if d.exists() else 0
            )
        plants = {c.split("___")[0] for c in manager.class_names}
        return {
            "available": counts["train"] > 0,
            "train_images": counts["train"],
            "valid_images": counts["valid"],
            "test_images": counts["test"],
            "total_images": sum(counts.values()),
            "total_classes": len(manager.class_names),
            "plant_types": len(plants),
            "healthy_classes": sum(c.lower().endswith("healthy") for c in manager.class_names),
        }

    # ------------------------------------------------------------------ frontend
    @app.route("/")
    def index():
        return send_from_directory(config.FRONTEND_DIR, "index.html")

    @app.route("/analyze")
    @app.route("/analyze.html")
    def analyze_page():
        return send_from_directory(config.FRONTEND_DIR, "analyze.html")

    @app.route("/stats")
    @app.route("/stats.html")
    def stats_page():
        return send_from_directory(config.FRONTEND_DIR, "stats.html")

    @app.route("/frontend/<path:path>")
    @app.route("/<path:path>")
    def static_files(path):
        target = (config.FRONTEND_DIR / path).resolve()
        if not str(target).startswith(str(config.FRONTEND_DIR)) or not target.is_file():
            return _error("Not found", 404)
        return send_from_directory(config.FRONTEND_DIR, path)

    @app.route("/api/uploads/<path:name>")
    def uploaded_file(name):
        return send_from_directory(config.UPLOAD_DIR, secure_filename(name))

    # ------------------------------------------------------------------ meta
    @app.route("/api/health")
    @app.route("/api")
    def health():
        return jsonify(
            {
                "status": "online",
                "version": config.APP_VERSION,
                "device": str(manager.device),
                "models_loaded": list(manager.models),
                "models_trained": manager.trained_models(),
                "field_model": field.available,
                "total_classes": len(manager.class_names),
                "timestamp": _now(),
            }
        )

    @app.route("/api/models")
    def list_models():
        return jsonify(
            {
                "success": True,
                "models": manager.all_model_info(),
                "best_model": manager.best_model(),
                "ensemble": manager.metrics.get("ensemble"),
                "field_model": field.describe(),
                "timestamp": _now(),
            }
        )

    @app.route("/api/classes")
    def list_classes():
        classes = [{"id": i, "name": n, **describe_class(n)} for i, n in enumerate(manager.class_names)]
        grouped: dict[str, list] = {}
        for c in classes:
            grouped.setdefault(c["plant"], []).append(c)
        return jsonify({"success": True, "classes": classes, "total": len(classes), "grouped_by_plant": grouped})

    @app.route("/api/dataset_info")
    def get_dataset_info():
        return jsonify(
            {"success": True, **dataset_info(), "model_architectures": len(manager.models), "timestamp": _now()}
        )

    @app.route("/api/stats")
    def stats():
        return jsonify(
            {
                "success": True,
                "statistics": manager.metrics,
                "models": manager.all_model_info(),
                "best_model": manager.best_model(),
                "training_history": {n: validation.load_training_history(n) for n in manager.models},
                "dataset": dataset_info(),
                "field_model": field.describe(),
                "total_classes": len(manager.class_names),
                "timestamp": _now(),
            }
        )

    @app.route("/api/validation/all")
    def validation_all():
        lang = _lang()
        reports = {n: validation.build_report(n, lang) for n in manager.models}
        return jsonify({"success": True, "reports": {k: v for k, v in reports.items() if v}, "timestamp": _now()})

    @app.route("/api/validation/<model_name>")
    def validation_one(model_name):
        report = validation.build_report(model_name, _lang())
        if not report:
            return _error(f"No validation data for '{model_name}'. Run scripts/evaluate_models.py.", 404)
        return jsonify({"success": True, "report": report, "timestamp": _now()})

    @app.route("/api/confusion_matrix/<model_name>")
    def confusion_matrix(model_name):
        report = validation.build_report(model_name, _lang())
        if not report:
            return _error(f"No validation data for '{model_name}'", 404)
        return jsonify(
            {
                "success": True,
                "model_name": model_name,
                "labels": report["labels"],
                "class_names": report["class_names"],
                "confusion_matrix": report["confusion_matrix"],
                "top_confused_pairs": report["top_confused_pairs"],
            }
        )

    @app.route("/api/research")
    def research():
        """Domain adaptation study summary (written by domain_adaptation_experiments/da/run_all.py)."""
        summary_path = config.DA_RESULTS_DIR / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
        runs = sorted(
            p.stem for p in config.DA_RESULTS_DIR.glob("*.json") if not p.stem.startswith(("splits_", "summary"))
        )
        return jsonify(
            {
                "success": True,
                "available": summary is not None,
                "summary": summary,
                "runs": runs,
                "field_model": field.describe(),
                "timestamp": _now(),
            }
        )

    @app.route("/api/treatment/<path:disease_class>")
    def treatment(disease_class):
        return jsonify(
            {"success": True, "disease_class": disease_class, "treatment": get_treatment(disease_class, _lang())}
        )

    # ------------------------------------------------------------------ inference
    @app.route("/api/upload", methods=["POST"])
    def upload():
        files = request.files.getlist("files") or request.files.getlist("file")
        if not files or all(not f.filename for f in files):
            return _error("No files provided")
        uploaded, errors = [], []
        for f in files:
            ext = f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else ""
            if ext not in config.ALLOWED_EXTENSIONS:
                errors.append(f"{f.filename}: unsupported extension")
                continue
            safe = secure_filename(f.filename) or f"image.{ext}"
            name = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}_{safe}"
            f.save(config.UPLOAD_DIR / name)
            try:
                load_image(config.UPLOAD_DIR / name).verify()
            except Exception:
                (config.UPLOAD_DIR / name).unlink(missing_ok=True)
                errors.append(f"{f.filename}: not a valid image")
                continue
            uploaded.append({"original_name": f.filename, "saved_name": name, "url": f"/api/uploads/{name}"})
        return jsonify({"success": bool(uploaded), "uploaded": uploaded, "errors": errors, "count": len(uploaded)})

    @app.route("/api/predict", methods=["POST"])
    def predict():
        data = request.get_json(silent=True) or {}
        try:
            path = resolve_image(data.get("image_path") or "")
        except (FileNotFoundError, PermissionError) as e:
            return _error(str(e), 404)
        except ValueError as e:
            return _error(str(e))
        try:
            result = run_prediction(
                path,
                data.get("model", "efficientnet"),
                data.get("explanation", "gradcam"),
                data.get("dataset_type", "controlled"),
                _lang(),
                data.get("ensemble_method"),
            )
        except KeyError as e:
            return _error(str(e).strip("'"), 400, available_models=list(manager.models))
        except ValueError as e:
            return _error(str(e))
        return jsonify(result)

    @app.route("/api/ensemble", methods=["POST"])
    def ensemble():
        data = request.get_json(silent=True) or {}
        try:
            path = resolve_image(data.get("image_path") or "")
        except (FileNotFoundError, PermissionError) as e:
            return _error(str(e), 404)
        except ValueError as e:
            return _error(str(e))
        result = run_prediction(
            path,
            "ensemble",
            data.get("explanation", "gradcam"),
            data.get("dataset_type", "controlled"),
            _lang(),
            data.get("ensemble_method", "weighted"),
        )
        return jsonify(result)

    @app.route("/api/compare", methods=["POST"])
    def compare():
        data = request.get_json(silent=True) or {}
        try:
            path = resolve_image(data.get("image_path") or "")
        except (FileNotFoundError, PermissionError) as e:
            return _error(str(e), 404)
        except ValueError as e:
            return _error(str(e))
        names = [n for n in (data.get("models") or list(manager.models)) if n in manager.models]
        if not names:
            return _error("None of the requested models are loaded", 400, available_models=list(manager.models))
        tensor = to_tensor(load_image(path))
        results = {}
        for n in names:
            t0 = time.time()
            pred = manager.predict(n, tensor)
            elapsed = (time.time() - t0) * 1000
            p, top = format_prediction(pred["probabilities"], pred["predicted_idx"], top_k=3)
            results[n] = {
                **p,
                "label": manager.model_info(n)["label"],
                "trained": manager.is_trained(n),
                "inference_time_ms": round(elapsed, 1),
                "top_predictions": top,
            }
        predicted = [r["class_raw"] for r in results.values()]
        majority = max(set(predicted), key=predicted.count)
        return jsonify(
            {
                "success": True,
                "image_name": path.name,
                "image_url": f"/api/uploads/{path.name}",
                "comparisons": results,
                "agreement": {
                    "class": describe_class(majority),
                    "models_agreeing": predicted.count(majority),
                    "models_total": len(predicted),
                },
                "timestamp": _now(),
            }
        )

    @app.route("/api/batch", methods=["POST"])
    def batch():
        data = request.get_json(silent=True) or {}
        paths = data.get("image_paths") or []
        if not paths:
            return _error("image_paths is required")
        results = []
        for name in paths:
            try:
                path = resolve_image(name)
                results.append(
                    run_prediction(
                        path,
                        data.get("model", "efficientnet"),
                        data.get("explanation", "gradcam"),
                        data.get("dataset_type", "controlled"),
                        _lang(),
                    )
                )
            except Exception as e:
                results.append({"success": False, "image_name": name, "error": str(e)})
        return jsonify({"success": True, "results": results, "total": len(results), "timestamp": _now()})

    # ------------------------------------------------------------------ export
    def _send(data: bytes, mimetype: str, filename: str):
        return send_file(io.BytesIO(data), mimetype=mimetype, as_attachment=True, download_name=filename)

    @app.route("/api/export/<fmt>", methods=["POST"])
    def export_results(fmt):
        data = request.get_json(silent=True) or {}
        results = data.get("results") or []
        if isinstance(results, dict):
            results = [results]
        if not results:
            return _error("results is required")
        lang = _lang()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if fmt == "csv":
            return _send(
                export_utils.results_to_csv(results, lang).encode("utf-8-sig"), "text/csv", f"crop_analysis_{stamp}.csv"
            )
        if fmt == "json":
            return _send(
                export_utils.results_to_json(results).encode("utf-8"), "application/json", f"crop_analysis_{stamp}.json"
            )
        if fmt in ("excel", "xlsx"):
            return _send(
                export_utils.results_to_excel(results, lang),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                f"crop_analysis_{stamp}.xlsx",
            )
        if fmt == "pdf":
            return _send(create_pdf_report(results, lang), "application/pdf", f"crop_analysis_{stamp}.pdf")
        return _error(f"Unknown export format: {fmt}", 404)

    @app.route("/api/export/comparison/<fmt>", methods=["POST"])
    def export_comparison(fmt):
        data = request.get_json(silent=True) or {}
        if not data.get("comparisons"):
            return _error("comparisons is required")
        lang = _lang()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if fmt == "csv":
            return _send(
                export_utils.comparison_to_csv(data, lang).encode("utf-8-sig"),
                "text/csv",
                f"model_comparison_{stamp}.csv",
            )
        if fmt in ("excel", "xlsx"):
            return _send(
                export_utils.comparison_to_excel(data, lang),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                f"model_comparison_{stamp}.xlsx",
            )
        if fmt == "pdf":
            return _send(create_comparison_report(data, lang), "application/pdf", f"model_comparison_{stamp}.pdf")
        return _error(f"Unknown export format: {fmt}", 404)

    @app.route("/api/generate_report", methods=["POST"])
    def generate_report():  # backwards compatible alias
        data = request.get_json(silent=True) or {}
        results = data.get("results") or [data]
        return _send(
            create_pdf_report(results, _lang()), "application/pdf", f"crop_analysis_{datetime.now():%Y%m%d_%H%M%S}.pdf"
        )

    # ------------------------------------------------------------------ errors
    @app.errorhandler(413)
    def too_large(e):
        return _error("File too large", 413, max_size_mb=config.MAX_FILE_SIZE // (1024 * 1024))

    @app.errorhandler(404)
    def not_found(e):
        return _error("Not found", 404)

    @app.errorhandler(Exception)
    def unhandled(e):
        app.logger.exception("Unhandled error")
        payload = {"success": False, "error": str(e) or e.__class__.__name__}
        if app.debug:
            payload["traceback"] = traceback.format_exc()
        return jsonify(payload), 500

    return app


app = create_app()


if __name__ == "__main__":
    print("=" * 60)
    print(f"Crop Disease Detection API v{config.APP_VERSION}")
    print(f"  models dir : {config.MODELS_DIR}")
    print(f"  uploads    : {config.UPLOAD_DIR}")
    print(f"  device     : {app.extensions['manager'].device}")
    print(f"  models     : {list(app.extensions['manager'].models)}")
    print(f"  open       : http://localhost:{config.PORT}/")
    print("=" * 60)
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG, threaded=True)
