"""Validación cruzada estratificada (k-fold) sin fuga de datos.

Reglas que garantizan la ausencia de *data leakage*:

1. El conjunto de TEST no participa: la validación cruzada se hace solo sobre el
   conjunto de desarrollo (Train + Validación de la partición 70/15/15).
2. En cada fold se construye un pipeline NUEVO (preprocesamiento + modelo) y se
   ajusta únicamente con las filas de entrenamiento de ese fold. El imputador, el
   escalador y el One-Hot nunca ven las filas de validación del fold.
3. Los folds son estratificados: cada uno conserva la proporción de venenosos.
"""
from time import perf_counter
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.models import create_model_pipeline, get_candidate_models
from src.utils import evaluate_thresholds, find_zero_fn_threshold

METRICAS_CV = ["Accuracy", "Precision", "Recall (Poisonous)", "F1-Score", "ROC-AUC"]
REJILLA_UMBRALES = np.round(np.linspace(0.01, 0.95, 95), 2)


def folds_estratificados(X: pd.DataFrame, y: pd.Series, k: int = 5, random_state: int = 42):
    """Índices (entrenamiento, validación) de cada fold, estratificados y reproducibles."""
    return list(StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state).split(X, y))


def _metricas_con_umbral(y_true: np.ndarray, proba: np.ndarray, umbral: float) -> dict:
    pred = (proba >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "Accuracy": (tp + tn) / len(y_true),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall (Poisonous)": recall_score(y_true, pred, zero_division=0),
        "F1-Score": f1_score(y_true, pred, zero_division=0),
        "FN": int(fn),
        "FP": int(fp),
    }


def validacion_cruzada_modelo(
    nombre_modelo: str,
    X: pd.DataFrame,
    y: pd.Series,
    columnas_numericas: list,
    columnas_categoricas: list,
    k: int = 5,
    tau: float = 0.5,
    random_state: int = 42,
    al_terminar_fold: Optional[Callable[[int, int], None]] = None,
    devolver_oof: bool = False,
):
    """Ejecuta k-fold estratificado para un modelo candidato y devuelve una fila por fold.

    Métricas de clasificación con el umbral por defecto 0.5; además FN/FP con el umbral
    de riesgo cero `tau` y el τ* que se habría calibrado dentro de cada fold.
    Con `devolver_oof=True` devuelve también P(venenoso) fuera de fold de cada fila de X
    (cada fila predicha por el modelo que no la vio al entrenar).
    """
    estimador_base = get_candidate_models()[nombre_modelo]
    y = pd.Series(np.asarray(y).astype(int), index=X.index)
    filas = []
    proba_oof = np.full(len(X), np.nan)
    folds = folds_estratificados(X, y, k=k, random_state=random_state)
    for numero, (idx_ent, idx_val) in enumerate(folds, start=1):
        # Pipeline nuevo en cada fold: el preprocesamiento se ajusta SOLO con idx_ent.
        pipeline = create_model_pipeline(clone(estimador_base), columnas_numericas, columnas_categoricas)
        inicio = perf_counter()
        pipeline.fit(X.iloc[idx_ent], y.iloc[idx_ent])
        segundos = perf_counter() - inicio

        y_val = y.iloc[idx_val].to_numpy()
        proba = pipeline.predict_proba(X.iloc[idx_val])[:, 1]
        proba_oof[idx_val] = proba

        por_defecto = _metricas_con_umbral(y_val, proba, 0.5)
        con_tau = _metricas_con_umbral(y_val, proba, tau)
        tau_fold, _ = find_zero_fn_threshold(evaluate_thresholds(y_val, proba, REJILLA_UMBRALES))

        filas.append({
            "Model": nombre_modelo,
            "Fold": numero,
            "n entrenamiento": len(idx_ent),
            "n validación": len(idx_val),
            "% venenosos (validación)": float(y_val.mean()),
            **{m: por_defecto[m] for m in METRICAS_CV if m != "ROC-AUC"},
            "ROC-AUC": float(roc_auc_score(y_val, proba)) if len(np.unique(y_val)) > 1 else np.nan,
            "FN (0.5)": por_defecto["FN"],
            "FP (0.5)": por_defecto["FP"],
            "FN (τ*)": con_tau["FN"],
            "FP (τ*)": con_tau["FP"],
            "τ* del fold": tau_fold,
            "Tiempo (s)": round(segundos, 2),
        })
        if al_terminar_fold is not None:
            al_terminar_fold(numero, k)
    resultados = pd.DataFrame(filas)
    return (resultados, proba_oof) if devolver_oof else resultados


def resumen_validacion_cruzada(resultados: pd.DataFrame) -> pd.DataFrame:
    """Media y desviación estándar de cada métrica por modelo, más los totales de errores."""
    agrupado = resultados.groupby("Model", sort=False)
    resumen = pd.DataFrame(index=agrupado.size().index)
    for metrica in METRICAS_CV:
        resumen[f"{metrica} (media)"] = agrupado[metrica].mean()
        resumen[f"{metrica} (desv.)"] = agrupado[metrica].std(ddof=1)
    resumen["FN totales (0.5)"] = agrupado["FN (0.5)"].sum()
    resumen["FN totales (τ*)"] = agrupado["FN (τ*)"].sum()
    resumen["Folds con FN = 0 (τ*)"] = agrupado["FN (τ*)"].apply(lambda s: f"{int((s == 0).sum())}/{len(s)}")
    resumen["τ* mínimo entre folds"] = agrupado["τ* del fold"].min()
    resumen["Tiempo medio (s)"] = agrupado["Tiempo (s)"].mean()
    return resumen.reset_index()


def filas_repetidas_entre(referencia: pd.DataFrame, evaluacion: pd.DataFrame) -> int:
    """Cuántas filas de `evaluacion` son idénticas (en todas las columnas) a alguna de `referencia`.

    Un valor > 0 indica fuga por duplicados: el modelo puede «recordar» esas filas.
    """
    columnas = list(referencia.columns)
    huella_ref = set(pd.util.hash_pandas_object(referencia[columnas], index=False))
    huella_eval = pd.util.hash_pandas_object(evaluacion[columnas], index=False)
    return int(huella_eval.isin(huella_ref).sum())


def esquema_folds(k: int) -> Iterable[list]:
    """Matriz k×k para dibujar el esquema: 1 = bloque de validación, 0 = entrenamiento."""
    return [[1 if fila == col else 0 for col in range(k)] for fila in range(k)]
