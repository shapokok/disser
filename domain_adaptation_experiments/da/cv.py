#!/usr/bin/env python3
"""
5-fold cross-validation of the domain adaptation methods (the thesis' main protocol).

All PlantDoc images of the 4 focus classes (train + test, 500 images) are split into 5
stratified folds. For every fold the other 4 folds are the target-domain training part:
85 % of it is used for adaptation, 15 % ("inner dev") only for early stopping of the
supervised methods. The held-out fold is predicted exactly once, so every image receives
one out-of-fold prediction -> metrics on n = 500 instead of 126, no selection bias.

    python -m da.cv --device mps --seeds 42 43 44                 # all methods
    python -m da.cv --methods adabn joint_v2 --seeds 42
    python -m da.cv --final joint_v2 --device mps                 # train the deployed field model

Methods
    zero_shot         PlantVillage MobileNet-V2 as is
    adabn             AdaBN: BatchNorm statistics re-estimated on unlabelled field images
    tent              TENT: entropy minimisation of BN affine parameters (unlabelled), after AdaBN
    self_training     the original self-training recipe (threshold 0.85 -> 0.65, balanced sampling)
    self_training_v2  CBST class-balanced selection + EMA teacher + strong augmentation + source replay
    joint             the original supervised joint fine-tuning recipe
    joint_v2          AdaBN init + field augmentation + PlantVillage replay + EMA weights (+ "_tta" variant)
    joint_v2_noadabn  the same without AdaBN (ablation)
    semi25            25 % of the labels + CBST pseudo-labels on the rest (and "sup25": the 25 % alone)

Unsupervised methods (adabn, tent, self_training*) never see target labels; their inner-dev
split is used as unlabelled data. Class-set knowledge (the 4 target classes) is assumed by the
pseudo-labelling methods, as in partial domain adaptation.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader

from da import common as C

CV_DIR = C.RESULTS_DIR / "cv"
K_FOLDS = 5
INNER_DEV = 0.15
FOLD_SEED = 42  # the fold assignment is fixed; --seeds only change training randomness
FOCUS = C.FOCUS_IDX

# --------------------------------------------------------------------------- image cache (workers=0, fast epochs)
_CACHE: dict[str, Image.Image] = {}


def _load(path) -> Image.Image:
    key = str(path)
    img = _CACHE.get(key)
    if img is None:
        img = Image.open(path).convert("RGB")
        img.thumbnail((448, 448))
        _CACHE[key] = img
    return img


class CachedTarget(C.PlantDocDataset):
    def __getitem__(self, i):
        rel, label = self.items[i]
        label = self.pseudo_labels.get(rel, label)
        img = _load(C.PLANTDOC_DIR / rel)
        return (self.transform(img) if self.transform else img), label, i


class CachedSource(C.PathDataset):
    def __getitem__(self, i):
        path, label = self.items[i]
        return self.transform(_load(path)), label, i


def target_loader(items, transform, batch, shuffle=False, balanced=False, pseudo=None):
    ds = CachedTarget(items, transform, pseudo)
    sampler = None
    if balanced and len(ds):
        counts = {}
        for lbl in ds.labels:
            counts[lbl] = counts.get(lbl, 0) + 1
        weights = [1.0 / counts[lbl] for lbl in ds.labels]
        sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(ds), replacement=True)
        shuffle = False
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, sampler=sampler, num_workers=0)


@torch.no_grad()
def predict_items(model, items, device, transform=None, batch=64):
    return C.predict(model, target_loader(items, transform or C.eval_transform(), batch), device)


def open_acc(model, items, device) -> float:
    probs, labels = predict_items(model, items, device)
    return float((probs.argmax(1) == labels).mean())


def restricted_acc(model, items, device) -> float:
    probs, labels = predict_items(model, items, device)
    return float((C.restricted_argmax(probs, FOCUS) == labels).mean())


class use_img_size:
    """Temporarily change the input resolution used by every transform (C.IMG_SIZE)."""

    def __init__(self, size: int):
        self.size = size

    def __enter__(self):
        self.saved, C.IMG_SIZE = C.IMG_SIZE, self.size

    def __exit__(self, *exc):
        C.IMG_SIZE = self.saved


_FOCUS_MAP = None


def _to_focus(y):
    global _FOCUS_MAP
    if _FOCUS_MAP is None or _FOCUS_MAP.device != y.device:
        m = torch.full((len(C.CLASS_NAMES),), -100, dtype=torch.long)
        for k, c in enumerate(FOCUS):
            m[c] = k
        _FOCUS_MAP = m.to(y.device)
    return _FOCUS_MAP[y]


# --------------------------------------------------------------------------- training loop
class Ctx:
    def __init__(self, device, seed, replay_items):
        self.device, self.seed, self.replay_items = device, seed, replay_items


def train(
    model,
    items,
    ctx: Ctx,
    *,
    epochs: int,
    lr: float,
    augment: str = "field",
    balanced: bool = True,
    replay: int = 0,
    ema: C.EMA | None = None,
    ema_decay: float | None = None,
    inner=None,
    pseudo=None,
    label_smoothing: float = 0.05,
    batch: int = 32,
    closed: bool = False,
    log: str = "",
):
    """Supervised / pseudo-label fine-tuning with optional source replay, EMA weights and early stopping."""
    device = ctx.device
    loader = target_loader(items, C.AUGMENTATIONS[augment](), batch, shuffle=True, balanced=balanced, pseudo=pseudo)
    src = None
    if replay:
        src_ds = CachedSource(ctx.replay_items, C.light_augment())
        src = C.infinite(DataLoader(src_ds, batch_size=replay, shuffle=True, num_workers=0, drop_last=True))
    if ema is None and ema_decay:
        ema = C.EMA(model, ema_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    total = max(1, epochs * len(loader))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    best = {"acc": -1.0, "epoch": 0, "state": None}
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        t0, loss_sum, n = time.time(), 0.0, 0
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out[:, FOCUS], _to_focus(y)) if closed else criterion(out, y)
            if src is not None:
                xs, ys, _ = next(src)
                with C.bn_frozen_stats(model):
                    loss = loss + criterion(model(xs.to(device)), ys.to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if ema is not None:
                ema.update(model)
            loss_sum += loss.item() * y.size(0)
            n += y.size(0)
        evaluated = ema.model if ema is not None else model
        row = {"epoch": epoch, "loss": loss_sum / max(1, n), "seconds": round(time.time() - t0, 1)}
        if inner:
            row["inner_acc"] = (restricted_acc if closed else open_acc)(evaluated, inner, device)
            if row["inner_acc"] > best["acc"]:
                best = {"acc": row["inner_acc"], "epoch": epoch, "state": copy.deepcopy(evaluated.state_dict())}
        history.append(row)
        if log and (epoch % 5 == 0 or epoch == epochs):
            extra = f" inner {row['inner_acc'] * 100:.1f}%" if inner else ""
            print(f"{log} epoch {epoch}/{epochs} loss {row['loss']:.3f}{extra} ({row['seconds']}s)", flush=True)
    result = ema.model if ema is not None else model
    if inner and best["state"] is not None:
        result.load_state_dict(best["state"])
    return result.eval(), {"history": history, "best_epoch": best["epoch"], "best_inner": best["acc"]}, ema


def cbst_select(teacher, pool, device, proportion: float, true_labels: bool = True):
    """Class-balanced self-training selection (Zou et al., 2018): the most confident `proportion` per class."""
    probs, labels = predict_items(teacher, pool, device)
    cs = C.closed_set_probs(probs, FOCUS)
    pred, conf = cs.argmax(1), cs.max(1)
    chosen = []
    for k in range(len(FOCUS)):
        idx = np.where(pred == k)[0]
        if len(idx) == 0:
            continue
        top = idx[np.argsort(-conf[idx])[: max(1, math.ceil(proportion * len(idx)))]]
        chosen += top.tolist()
    pseudo = {pool[i][0]: FOCUS[pred[i]] for i in chosen}
    acc = float(np.mean([FOCUS[pred[i]] == labels[i] for i in chosen])) if chosen and true_labels else None
    return pseudo, {"selected": len(chosen), "pseudo_label_accuracy": acc}


# --------------------------------------------------------------------------- methods
def m_zero_shot(ctx, fit, inner, info):
    return {"zero_shot": C.load_source_model(ctx.device)}


def m_adabn(ctx, fit, inner, info):
    return {"adabn": C.adapt_bn(C.load_source_model(ctx.device), fit + inner, ctx.device)}


def m_tent(ctx, fit, inner, info, epochs: int = 1, lr: float = 1e-3):
    model = C.adapt_bn(C.load_source_model(ctx.device), fit + inner, ctx.device)
    model.eval()
    params = []
    for p in model.parameters():
        p.requires_grad_(False)
    for m in C.bn_layers(model):
        m.train()
        for p in (m.weight, m.bias):
            p.requires_grad_(True)
            params.append(p)
    opt = torch.optim.Adam(params, lr=lr)
    loader = target_loader(fit + inner, C.eval_transform(), 32, shuffle=True)
    for _ in range(epochs):
        for x, _, _ in loader:
            p = F.softmax(model(x.to(ctx.device)), 1)
            loss = -(p * torch.log(p.clamp_min(1e-8))).sum(1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    for p in model.parameters():
        p.requires_grad_(True)
    return {"tent": model.eval()}


def m_self_training(ctx, fit, inner, info):
    """The original recipe (for comparison): raw 38-way confidence thresholds, balanced oversampling."""
    model = C.load_source_model(ctx.device)
    pool = fit + inner
    rounds = []
    for it in range(5):
        thr = 0.85 - 0.05 * it
        probs, labels = predict_items(model, pool, ctx.device)
        focus = probs[:, FOCUS]
        conf, pred = focus.max(1), np.array(FOCUS)[focus.argmax(1)]
        keep = np.where(conf >= thr)[0]
        if len(keep) < 32:
            break
        pseudo = {pool[i][0]: int(pred[i]) for i in keep}
        rounds.append(
            {
                "threshold": thr,
                "selected": int(len(keep)),
                "pseudo_label_accuracy": float((pred[keep] == labels[keep]).mean()),
            }
        )
        model, _, _ = train(
            model,
            [pool[i] for i in keep],
            ctx,
            epochs=3,
            lr=1e-4,
            augment="light",
            balanced=True,
            pseudo=pseudo,
            label_smoothing=0.0,
        )
    info["self_training"] = {"rounds": rounds}
    return {"self_training": model}


def _cbst_rounds(model, ema, unlabeled, labeled, ctx, schedule, inner=None, log=""):
    """Shared CBST loop: the EMA teacher selects pseudo-labels, the student trains on labeled + selected."""
    rounds, best = [], {"acc": -1.0, "state": None, "round": 0}
    for r, prop in enumerate(schedule, 1):
        pseudo, diag = cbst_select(ema.model, unlabeled, ctx.device, prop)
        items = labeled + [it for it in unlabeled if it[0] in pseudo]
        _, _, ema = train(
            model,
            items,
            ctx,
            epochs=3,
            lr=1e-4,
            augment="field",
            balanced=False,
            replay=32,
            ema=ema,
            pseudo=pseudo,
            label_smoothing=0.1,
        )
        diag["proportion"] = prop
        if inner:
            diag["inner_acc"] = open_acc(ema.model, inner, ctx.device)
            if diag["inner_acc"] > best["acc"]:
                best = {"acc": diag["inner_acc"], "state": copy.deepcopy(ema.model.state_dict()), "round": r}
        rounds.append(diag)
        if log:
            pla = diag["pseudo_label_accuracy"]
            print(
                f"{log} round {r}: p={prop:.2f} selected {diag['selected']} (pseudo-label acc {pla * 100:.1f}%, diagnostic)",
                flush=True,
            )
    if inner and best["state"] is not None:
        ema.model.load_state_dict(best["state"])
    return ema.model.eval(), rounds


def m_self_training_v2(ctx, fit, inner, info, log=""):
    student = C.load_source_model(ctx.device)
    ema = C.EMA(student, decay=0.98)
    model, rounds = _cbst_rounds(student, ema, fit + inner, [], ctx, [0.2, 0.35, 0.5, 0.65, 0.8], log=log)
    info["self_training_v2"] = {"rounds": rounds}
    return {"self_training_v2": model}


def joint_recipe(ctx, fit, inner, *, arch="mobilenet", epochs=25, lr=1e-4, closed=False, log=""):
    """The original supervised recipe: heavy augmentation, class-balanced sampling, early stopping on inner dev."""
    return train(
        C.load_source_model(ctx.device, arch),
        fit,
        ctx,
        epochs=epochs,
        lr=lr,
        augment="heavy",
        balanced=True,
        inner=inner,
        label_smoothing=0.05,
        closed=closed,
        log=log,
    )


def _joint_variant(key, img_size=None, **kw):
    def method(ctx, fit, inner, info, log=""):
        with use_img_size(img_size or C.IMG_SIZE):
            model, meta, _ = joint_recipe(ctx, fit, inner, log=log, **kw)
        model._img_size = img_size or C.IMG_SIZE
        info[key] = {k: meta[k] for k in ("best_epoch", "best_inner")}
        return {key: model}

    return method


m_joint = _joint_variant("joint")
m_joint_long = _joint_variant("joint_long", epochs=60)
m_joint_effnet = _joint_variant("joint_effnet", arch="efficientnet", epochs=40)
m_joint_320 = _joint_variant("joint_320", img_size=320)
m_joint_closed = _joint_variant("joint_closed", closed=True)


def joint_v2_recipe(ctx, labeled, inner, unlabeled_for_bn, log="", epochs=30, adabn=True):
    model = C.load_source_model(ctx.device)
    if adabn:
        model = C.adapt_bn(model, unlabeled_for_bn, ctx.device)
    return train(
        model,
        labeled,
        ctx,
        epochs=epochs,
        lr=2e-4,
        augment="field",
        balanced=True,
        replay=32,
        ema_decay=0.99,
        inner=inner,
        label_smoothing=0.1,
        log=log,
    )


def m_joint_v2(ctx, fit, inner, info, log=""):
    model, meta, _ = joint_v2_recipe(ctx, fit, inner, fit + inner, log=log)
    info["joint_v2"] = {k: meta[k] for k in ("best_epoch", "best_inner")}
    return {"joint_v2": model}


def m_joint_v2_noadabn(ctx, fit, inner, info, log=""):
    model, meta, _ = joint_v2_recipe(ctx, fit, inner, fit + inner, log=log, adabn=False)
    info["joint_v2_noadabn"] = {k: meta[k] for k in ("best_epoch", "best_inner")}
    return {"joint_v2_noadabn": model}


def m_semi25(ctx, fit, inner, info, log="", fraction: float = 0.25):
    labels = [lbl for _, lbl in fit]
    labeled, unlabeled = train_test_split(fit, train_size=fraction, stratify=labels, random_state=ctx.seed)
    sup, meta, _ = joint_v2_recipe(ctx, labeled, inner, fit + inner, log=log, adabn=False)
    sup_copy = copy.deepcopy(sup)
    for prm in sup.parameters():  # `sup` is an EMA copy (frozen); the student must be trainable again
        prm.requires_grad_(True)
    ema = C.EMA(sup, decay=0.98)
    semi, rounds = _cbst_rounds(sup, ema, unlabeled, labeled, ctx, [0.3, 0.5, 0.7, 0.9], inner=inner, log=log)
    info["semi25"] = {
        "labeled": len(labeled),
        "unlabeled": len(unlabeled),
        "rounds": rounds,
        "sup_best_epoch": meta["best_epoch"],
    }
    return {"sup25": sup_copy.eval(), "semi25": semi}


METHODS = {
    "zero_shot": m_zero_shot,
    "adabn": m_adabn,
    "tent": m_tent,
    "self_training": m_self_training,
    "self_training_v2": m_self_training_v2,
    "joint": m_joint,
    "joint_long": m_joint_long,
    "joint_effnet": m_joint_effnet,
    "joint_320": m_joint_320,
    "joint_closed": m_joint_closed,
    "joint_v2": m_joint_v2,
    "joint_v2_noadabn": m_joint_v2_noadabn,
    "semi25": m_semi25,
}
TTA_FOR = {"joint", "joint_long", "joint_effnet", "joint_320", "joint_closed", "joint_v2", "joint_v2_noadabn"}
LABELS = {
    "zero_shot": "Без адаптации",
    "adabn": "AdaBN",
    "tent": "AdaBN + TENT",
    "self_training": "Self-training (исходный)",
    "self_training_v2": "Self-training v2 (CBST + EMA-учитель)",
    "joint": "Joint fine-tuning (исходный)",
    "joint_tta": "Joint fine-tuning + TTA",
    "joint_long": "Joint fine-tuning, 60 эпох",
    "joint_long_tta": "Joint fine-tuning, 60 эпох + TTA",
    "joint_effnet": "Joint fine-tuning, EfficientNet-B0",
    "joint_effnet_tta": "Joint fine-tuning, EfficientNet-B0 + TTA",
    "joint_320": "Joint fine-tuning, 320 px",
    "joint_320_tta": "Joint fine-tuning, 320 px + TTA",
    "joint_closed": "Joint fine-tuning, 4-классовая функция потерь",
    "joint_closed_tta": "Joint fine-tuning, 4-классовая функция потерь + TTA",
    "joint_v2": "Joint fine-tuning v2",
    "joint_v2_tta": "Joint fine-tuning v2 + TTA",
    "joint_v2_noadabn": "Joint fine-tuning v2 без AdaBN",
    "joint_v2_noadabn_tta": "Joint fine-tuning v2 без AdaBN + TTA",
    "sup25": "25 % разметки",
    "semi25": "25 % разметки + псевдометки",
}
UNSUPERVISED = {"zero_shot", "adabn", "tent", "self_training", "self_training_v2"}


# --------------------------------------------------------------------------- CV driver
def pool_items():
    return [list(x) for x in C.list_images("train", C.FOCUS_CLASSES) + C.list_images("test", C.FOCUS_CLASSES)]


def make_folds(items):
    labels = [lbl for _, lbl in items]
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=FOLD_SEED)
    out = []
    for f, (tr, te) in enumerate(skf.split(np.zeros(len(items)), labels)):
        train_part = [items[i] for i in tr]
        fit, inner = train_test_split(
            train_part, test_size=INNER_DEV, stratify=[lbl for _, lbl in train_part], random_state=FOLD_SEED + f
        )
        out.append((f, fit, inner, [items[i] for i in te]))
    return out


@torch.no_grad()
def tta_predict(model, items, device):
    from da.tta import tta_views

    acc = None
    labels = None
    for tf in tta_views().values():
        probs, labels = predict_items(model, items, device, transform=tf)
        acc = probs if acc is None else acc + probs
    return acc / len(tta_views()), labels


def summarise_oof(name, seed, folds_out, config):
    labels = np.concatenate([f["labels"] for f in folds_out])
    probs = np.concatenate([f["probs"] for f in folds_out])
    per_fold = [float((f["probs"].argmax(1) == f["labels"]).mean()) for f in folds_out]
    return {
        "method": name,
        "label": LABELS.get(name, name),
        "seed": seed,
        "unsupervised": name in UNSUPERVISED,
        "protocol": {"k_folds": K_FOLDS, "inner_dev": INNER_DEV, "fold_seed": FOLD_SEED, "n": int(len(labels))},
        "config": config,
        "open": C.compute_metrics(labels, probs.argmax(1), FOCUS),
        "restricted": C.compute_metrics(labels, C.restricted_argmax(probs, FOCUS), FOCUS),
        "fold_accuracy": per_fold,
        "fold_accuracy_mean": float(np.mean(per_fold)),
        "fold_accuracy_std": float(np.std(per_fold)),
        "folds": [{k: v for k, v in f.items() if k not in ("labels", "probs", "paths")} for f in folds_out],
        "oof": {
            "paths": [p for f in folds_out for p in f["paths"]],
            "labels": labels.tolist(),
            "pred_open": probs.argmax(1).tolist(),
            "probs_focus": np.round(probs[:, FOCUS], 4).tolist(),
            "probs": np.round(probs, 4).tolist(),
        },
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }


def run_method(method, seed, folds, ctx_factory, config):
    outputs: dict[str, list] = {}
    t_start = time.time()
    for f, fit, inner, test in folds:
        C.set_seed(seed * 100 + f)
        ctx = ctx_factory(seed)
        info = {}
        t0 = time.time()
        models = METHODS[method](
            ctx,
            fit,
            inner,
            info,
            **(
                {"log": f"[{method} s{seed} f{f}]"}
                if method not in ("zero_shot", "adabn", "tent", "self_training")
                else {}
            ),
        )
        for name, model in models.items():
            with use_img_size(getattr(model, "_img_size", C.IMG_SIZE)):
                probs, labels = predict_items(model, test, ctx.device)
                variants = {name: probs}
                if name in TTA_FOR:
                    variants[name + "_tta"] = tta_predict(model, test, ctx.device)[0]
            for vname, vprobs in variants.items():
                outputs.setdefault(vname, []).append(
                    {
                        "fold": f,
                        "n_fit": len(fit),
                        "n_inner": len(inner),
                        "n_test": len(test),
                        "accuracy": float((vprobs.argmax(1) == labels).mean()),
                        "seconds": round(time.time() - t0, 1),
                        "info": info.get(name, info.get(method, {})),
                        "labels": labels,
                        "probs": vprobs,
                        "paths": [it[0] for it in test],
                    }
                )
        accs = "  ".join(f"{n}={o[-1]['accuracy'] * 100:.1f}%" for n, o in outputs.items())
        print(f"[{method} seed {seed}] fold {f}: {accs}  ({time.time() - t0:.0f}s)", flush=True)
    for name, folds_out in outputs.items():
        res = summarise_oof(name, seed, folds_out, config | {"seconds_total": round(time.time() - t_start, 1)})
        C.save_json(res, CV_DIR / f"{name}_seed{seed}.json")
        print(
            f"==> {name} seed {seed}: open {res['open']['accuracy'] * 100:.1f}% (CI {res['open']['ci95'][0] * 100:.0f}–{res['open']['ci95'][1] * 100:.0f}), restricted {res['restricted']['accuracy'] * 100:.1f}%, macro-F1 {res['open']['macro_f1'] * 100:.1f}",
            flush=True,
        )


def export_final(recipe: str, ctx_factory, seed: int):
    """Train the deployed field model on every PlantDoc image of the focus classes with the given recipe."""
    items = pool_items()
    fit, inner = train_test_split(
        items, test_size=INNER_DEV, stratify=[lbl for _, lbl in items], random_state=FOLD_SEED
    )
    C.set_seed(seed)
    ctx = ctx_factory(seed)
    info = {}
    model = METHODS[recipe](ctx, fit, inner, info, log=f"[final {recipe}]")[recipe]
    C.save_checkpoint(model, C.MODELS_DIR / "field_mobilenet_da.pth")
    cv = [json.loads(p.read_text()) for p in sorted(CV_DIR.glob(f"{recipe}_seed*.json"))]
    meta = {
        "method": LABELS.get(recipe, recipe),
        "method_key": recipe,
        "trained_on": f"all {len(items)} PlantDoc images of the 4 focus classes (15 % for early stopping)",
        "source_checkpoint": "mobilenet_model.pth",
        "covered_class_indices": FOCUS,
        "covered_classes": [C.CLASS_NAMES[i] for i in FOCUS],
        "accuracy_test": float(np.mean([r["open"]["accuracy"] for r in cv])) if cv else None,
        "accuracy_restricted": float(np.mean([r["restricted"]["accuracy"] for r in cv])) if cv else None,
        "evaluation": f"{K_FOLDS}-fold cross-validation, n = {cv[0]['protocol']['n']}, {len(cv)} seed(s)"
        if cv
        else None,
        "exported_at": datetime.now().isoformat(timespec="seconds"),
    }
    (C.MODELS_DIR / "field_model_info.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"exported {recipe} -> models/field_mobilenet_da.pth; CV accuracy {meta['accuracy_test']}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--replay-per-class", type=int, default=30)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--final", default=None, choices=list(METHODS), help="train and export the deployed field model")
    args = p.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    device = C.pick_device(args.device)
    replay_items = C.source_items(args.replay_per_class, seed=0)
    ctx_factory = lambda seed: Ctx(device, seed, replay_items)  # noqa: E731
    items = pool_items()
    folds = make_folds(items)
    print(
        f"device={device}  pool={len(items)}  folds={[len(t) for *_, t in folds]}  replay={len(replay_items)}",
        flush=True,
    )
    if args.final:
        export_final(args.final, ctx_factory, args.seeds[0])
        return
    config = {"device": str(device), "replay_per_class": args.replay_per_class}
    for seed in args.seeds:
        for method in args.methods:
            produced = (
                [method] + (["sup25"] if method == "semi25" else []) + ([method + "_tta"] if method in TTA_FOR else [])
            )
            if args.skip_existing and all((CV_DIR / f"{n}_seed{seed}.json").exists() for n in produced):
                print(f"-- skip {method} seed {seed}")
                continue
            run_method(method, seed, folds, ctx_factory, config)


if __name__ == "__main__":
    main()
