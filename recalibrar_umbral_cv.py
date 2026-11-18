# -*- coding: utf-8 -*-
"""Recalibra el umbral de riesgo cero τ* del modelo desplegado con validación cruzada.

El notebook 03 calibró τ = 0.67 con un único split de validación, tomando el umbral MÁS
ALTO con FN = 0. Esa regla tiene dos problemas:

* Depende del split: en la validación cruzada 5-fold, un fold deja pasar un venenoso con 0.67.
* Por construcción queda en el BORDE de la franja segura: el venenoso con menor probabilidad
  se salva solo porque la regla usa «mayor o igual». No deja margen.

Este script:

1. Reproduce la partición 70/15/15 (random_state=42) y toma Train + Validación como
   conjunto de desarrollo. El TEST no participa en la calibración.
2. Ejecuta k-fold estratificado con el mismo modelo del notebook 03 (pipeline nuevo por
   fold, sin fuga de datos) y guarda la probabilidad fuera de fold de cada registro.
3. Calcula la franja sin errores fuera de fold:
       a = máx P(venenoso) de un comestible,  b = mín P(venenoso) de un venenoso.
4. τ* = centro de la franja, (a + b) / 2, redondeado hacia abajo a 0.01 (el lado que
   protege de los falsos negativos). Así queda a la misma distancia de ambos errores.
   Si las clases se solapan (a ≥ b), se usa el umbral más alto con FN = 0 en todos los folds.
5. Verifica τ* en cada fold y en el test con el pipeline desplegado (solo lectura).
6. Escribe τ* en app/models/model_metadata.json junto con la trazabilidad del cálculo y
   regenera docs/figuras_modelo_matematico/ajuste_umbral_modelo_ganador.png.

Ejecutar después de notebooks/03_model_training.ipynb (que reescribe el metadata y la figura):
    python recalibrar_umbral_cv.py
"""
import json
import math
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data_loader import NUMERIC_COLUMNS, get_train_val_test_data, load_raw_data  # noqa: E402
from src.utils import evaluate_thresholds  # noqa: E402
from src.validation import REJILLA_UMBRALES, folds_estratificados, validacion_cruzada_modelo  # noqa: E402

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "app" / "models"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
FIGURA_UMBRAL = ROOT / "docs" / "figuras_modelo_matematico" / "ajuste_umbral_modelo_ganador.png"
K_FOLDS = 5
RANDOM_STATE = 42


def errores_con_umbral(y_true: np.ndarray, proba: np.ndarray, umbral: float) -> dict:
    pred = proba >= umbral
    tp = int((pred & (y_true == 1)).sum())
    fp = int((pred & (y_true == 0)).sum())
    fn = int((~pred & (y_true == 1)).sum())
    tn = int((~pred & (y_true == 0)).sum())
    return {
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
    }


def figura_umbral(barrido: pd.DataFrame, tau: float, tau_borde: float, franja: dict, titulo: str) -> None:
    """Sensibilidad, precisión y errores (FN y FP) según el umbral, con la franja sin errores."""
    a, b = franja["max_proba_edible"], franja["min_proba_poisonous"]
    fig, ax1 = plt.subplots(figsize=(9, 5))
    if a < b:
        ax1.axvspan(a, b, color="#c8e6c9", alpha=0.7, label=f"Franja sin errores ({a:.2f} – {b:.2f})")
    ax1.plot(barrido["threshold"], barrido["Recall (Poisonous)"], color="tab:green", lw=2.5,
             label="Recall (venenoso)")
    ax1.plot(barrido["threshold"], barrido["Precision"], color="tab:blue", lw=2, linestyle="--", label="Precisión")
    ax1.axvline(tau, color="black", lw=2, label=f"τ* desplegado = {tau:.2f} (centro de la franja)")
    if abs(tau_borde - tau) > 1e-9:
        ax1.axvline(tau_borde, color="#757575", lw=1.5, linestyle=":",
                    label=f"Umbral más alto con FN = 0 = {tau_borde:.2f} (borde)")
    ax1.set_xlabel("Umbral de decisión τ", fontsize=11)
    ax1.set_ylabel("Recall / Precisión", fontsize=11)
    ax1.set_ylim(-0.05, 1.05)

    ax2 = ax1.twinx()
    ax2.plot(barrido["threshold"], barrido["FN"], color="tab:red", lw=2, linestyle="-.", label="Falsos negativos (FN)")
    ax2.plot(barrido["threshold"], barrido["FP"], color="tab:orange", lw=2, linestyle=(0, (1, 1)),
             label="Falsos positivos (FP)")
    ax2.set_yscale("symlog", linthresh=1)
    ax2.set_ylabel("Número de errores (escala logarítmica)", fontsize=11)

    lineas = ax1.get_legend_handles_labels()
    lineas2 = ax2.get_legend_handles_labels()
    ax1.legend(lineas[0] + lineas2[0], lineas[1] + lineas2[1], loc="center left", fontsize=8.5, framealpha=0.92)
    ax1.set_title(titulo, fontsize=11.5, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    FIGURA_UMBRAL.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURA_UMBRAL, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    modelo = meta["model_name"]
    calibracion_previa = meta.get("threshold_calibration") or {}
    umbral_split_unico = float(calibracion_previa.get("single_split_threshold", meta["optimal_threshold"]))

    df = load_raw_data()
    X_tr, X_va, X_te, y_tr, y_va, y_te = get_train_val_test_data(df, random_state=RANDOM_STATE)
    X_dev = pd.concat([X_tr, X_va])
    y_dev = pd.concat([y_tr, y_va]).map({"e": 0, "p": 1}).astype(int)
    categoricas = [c for c in X_dev.columns if c not in NUMERIC_COLUMNS]

    print(f"Validación cruzada {K_FOLDS}-fold de {modelo} sobre {len(X_dev):,} especímenes de desarrollo...")
    folds, proba_oof = validacion_cruzada_modelo(
        modelo, X_dev, y_dev, NUMERIC_COLUMNS, categoricas, k=K_FOLDS, tau=umbral_split_unico,
        random_state=RANDOM_STATE, devolver_oof=True,
    )
    y_dev_np = y_dev.to_numpy()
    umbrales_fold = [round(float(t), 2) for t in folds["τ* del fold"]]
    tau_borde = min(umbrales_fold)

    # Franja sin errores con las probabilidades fuera de fold.
    a = float(proba_oof[y_dev_np == 0].max())
    b = float(proba_oof[y_dev_np == 1].min())
    franja = {"max_proba_edible": round(a, 4), "min_proba_poisonous": round(b, 4)}
    if a < b:
        tau = math.floor(round((a + b) / 2 * 100, 6)) / 100
        regla = "centro de la franja sin errores fuera de fold, redondeado hacia abajo a 0.01"
    else:
        tau = tau_borde
        regla = "umbral más alto con FN = 0 en todos los folds (las clases se solapan fuera de fold)"
    print(f"Franja sin errores fuera de fold: comestibles ≤ {a:.2f} y venenosos ≥ {b:.2f}")
    print(f"τ* = {tau:.2f} ({regla}); umbral de borde = {tau_borde:.2f}; antes = {umbral_split_unico:.2f}")

    # Errores de cada fold con τ*.
    fn_fold, fp_fold = [], []
    for numero, (_, idx_val) in enumerate(folds_estratificados(X_dev, y_dev, k=K_FOLDS, random_state=RANDOM_STATE), 1):
        errores = errores_con_umbral(y_dev_np[idx_val], proba_oof[idx_val], tau)
        fn_fold.append(errores["fn"])
        fp_fold.append(errores["fp"])
        print(f"  Fold {numero}: umbral más alto con FN = 0: {umbrales_fold[numero - 1]:.2f} | "
              f"con τ* = {tau:.2f}: FN = {errores['fn']}, FP = {errores['fp']}")

    barrido_oof = evaluate_thresholds(y_dev_np, proba_oof, REJILLA_UMBRALES)
    figura_umbral(barrido_oof, tau, tau_borde, franja,
                  f"Umbral de riesgo cero con validación cruzada {K_FOLDS}-fold — {modelo}")
    print(f"✓ Figura regenerada: {FIGURA_UMBRAL.relative_to(ROOT)}")

    # Verificación en TEST con el pipeline desplegado (no se usa para elegir τ*).
    pipeline = joblib.load(MODELS_DIR / "pipeline.joblib")
    y_test = y_te.map({"e": 0, "p": 1}).to_numpy()
    proba_test = pipeline.predict_proba(X_te[meta["numeric_features"] + meta["categorical_features"]])[:, 1]
    test_nuevo = errores_con_umbral(y_test, proba_test, tau)
    test_previo = errores_con_umbral(y_test, proba_test, umbral_split_unico)
    print(f"Test con τ* = {tau:.2f}: FN = {test_nuevo['fn']}, FP = {test_nuevo['fp']}, "
          f"precisión = {test_nuevo['precision']:.2%}, recall = {test_nuevo['recall']:.2%}")

    calibracion = {
        "rule": regla,
        "method": (
            f"Validación cruzada estratificada {K_FOLDS}-fold sobre Train + Validación "
            f"(random_state={RANDOM_STATE}), con un pipeline nuevo por fold. τ* es el centro de la franja "
            "sin errores de las probabilidades fuera de fold (entre el comestible con mayor P(venenoso) y el "
            "venenoso con menor P(venenoso)), redondeado hacia abajo a 0.01. El test no participa."
        ),
        "k_folds": K_FOLDS,
        "random_state": RANDOM_STATE,
        "cv_out_of_fold_band": franja,
        "margin_to_edible": round(tau - a, 4),
        "margin_to_poisonous": round(b - tau, 4),
        "edge_threshold": tau_borde,
        "fold_thresholds": umbrales_fold,
        "fold_fn_at_threshold": fn_fold,
        "fold_fp_at_threshold": fp_fold,
        "single_split_threshold": umbral_split_unico,
        "fold_fn_at_single_split_threshold": [int(v) for v in folds["FN (τ*)"]],
        "test_at_threshold": test_nuevo,
        "test_at_single_split_threshold": test_previo,
        "test_min_proba_poisonous": round(float(proba_test[y_test == 1].min()), 4),
        "test_max_proba_edible": round(float(proba_test[y_test == 0].max()), 4),
        "calibrated_at": datetime.now().isoformat(timespec="seconds"),
    }

    # Reescribe el metadata conservando el orden de las claves.
    nuevo_meta = {}
    for clave, valor in meta.items():
        if clave == "threshold_calibration":
            continue
        nuevo_meta[clave] = tau if clave == "optimal_threshold" else valor
        if clave == "optimal_threshold":
            nuevo_meta["threshold_calibration"] = calibracion
    METADATA_PATH.write_text(json.dumps(nuevo_meta, indent=4, ensure_ascii=False), encoding="utf-8")
    print(f"✓ {METADATA_PATH.relative_to(ROOT)} actualizado: optimal_threshold = {tau:.2f}")


if __name__ == "__main__":
    main()
