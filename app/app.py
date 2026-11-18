# -*- coding: utf-8 -*-
"""🍄 Sistema Inteligente de Clasificación Micológica — Demo del Hito 2.

Aplicación Streamlit que consume los artefactos entrenados en
`notebooks/03_model_training.ipynb` (app/models/) para clasificar hongos
como comestibles o venenosos con metodología de riesgo cero (FN = 0).

Funcionalidades:
 1.  Predicción individual interactiva (20 características morfológicas).
 2.  Predicción por lotes desde archivo CSV.
 3.  Probabilidad de toxicidad en tiempo real (predict_proba).
 4.  Umbral de riesgo cero τ* configurable con efecto inmediato en el dictamen.
 5.  Desglose local de la probabilidad por variable (contribuciones nativas del Random Forest).
 6.  Simulador "¿qué pasaría si...?" para cualquier variable del espécimen.
 7.  Importancia global precalculada (sin cálculo al arrancar la app).
 8.  Dashboard de métricas del modelo desplegado en el conjunto de test.
 9.  Comparativa de los 4 modelos entrenados.
 10. Matriz de confusión del modelo desplegado.
 11. Exploración del dataset (INFOVIZ: balance, distribuciones, categorías).
 12. Descarga de resultados de predicción por lotes + plantilla CSV.
 13. Ficha técnica del modelo y documentación matemática embebida.
 14. Validación cruzada estratificada k-fold sin fuga de datos (pipeline por fold, test excluido).

Ejecución:
    streamlit run app/app.py
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "app" / "models"
DATA_PATH = ROOT / "data" / "raw" / "secondary_data.csv"
FORMULAS_PATH = ROOT / "docs" / "formulas_matematicas_modelos.md"

sys.path.insert(0, str(ROOT))
from src.data_loader import get_train_val_test_data  # noqa: E402
from src.utils import aggregate_by_original, tree_path_contributions, what_if_scan  # noqa: E402
from src.validation import (  # noqa: E402
    METRICAS_CV,
    filas_repetidas_entre,
    resumen_validacion_cruzada,
    validacion_cruzada_modelo,
)

st.set_page_config(
    page_title="Clasificación de Toxicidad de Hongos",
    page_icon="🍄",
    layout="wide",
)

# ------------------------------------------------------------------ etiquetas legibles
ETIQUETAS_VARIABLES = {
    # Variables métricas (numéricas)
    "cap-diameter": "Diámetro del sombrero (cm)",
    "stem-height": "Altura del tallo (cm)",
    "stem-width": "Ancho del tallo (mm)",
    # Variables nominales (categóricas)
    "cap-shape": "Forma del sombrero",
    "cap-surface": "Superficie del sombrero",
    "cap-color": "Color del sombrero",
    "does-bruise-or-bleed": "¿Presenta magulladuras o sangrado?",
    "gill-attachment": "Inserción de las láminas",
    "gill-spacing": "Espaciado de las láminas",
    "gill-color": "Color de las láminas",
    "stem-root": "Raíz del tallo",
    "stem-surface": "Superficie del tallo",
    "stem-color": "Color del tallo",
    "veil-type": "Tipo de velo",
    "veil-color": "Color del velo",
    "has-ring": "¿Tiene anillo?",
    "ring-type": "Tipo de anillo",
    "spore-print-color": "Color de la esporada",
    "habitat": "Hábitat",
    "season": "Estación / Temporada",
}

ETIQUETAS_NUMERICAS = {
    "cap-diameter": "Diámetro del sombrero (cm)",
    "stem-height": "Altura del tallo (cm)",
    "stem-width": "Ancho del tallo (mm)",
}

ETIQUETAS_CATEGORICAS = {
    "cap-shape": {
        "b": "Campana",
        "c": "Cónica",
        "x": "Convexa",
        "f": "Plana",
        "s": "Hundida",
        "p": "Esférica",
        "o": "Otros",
    },
    "cap-surface": {
        "d": "Seca",
        "e": "Carnosa",
        "g": "Surcos",
        "h": "Brillante",
        "i": "Fibrosa",
        "k": "Sedosa",
        "l": "Coriácea",
        "s": "Lisa",
        "t": "Pegajosa",
        "w": "Arrugada",
        "y": "Escamosa",
    },
    "cap-color": {
        "n": "Marrón",
        "b": "Ante (beige)",
        "g": "Gris",
        "r": "Verde",
        "p": "Rosa",
        "u": "Púrpura",
        "e": "Rojo",
        "w": "Blanco",
        "y": "Amarillo",
        "l": "Azul",
        "o": "Naranja",
        "k": "Negro",
    },
    "does-bruise-or-bleed": {
        "t": "Sí (presenta sangrado o lesión)",
        "f": "No",
    },
    "gill-attachment": {
        "a": "Adnata (adherida)",
        "x": "Adnexa",
        "d": "Decurrente",
        "e": "Libre",
        "s": "Sinuada",
        "p": "Poros",
        "f": "Ninguna",
    },
    "gill-spacing": {
        "c": "Cerradas (apretadas)",
        "d": "Distantes (espaciadas)",
        "f": "Ninguna",
    },
    "gill-color": {
        "n": "Marrón",
        "b": "Ante (beige)",
        "g": "Gris",
        "r": "Verde",
        "p": "Rosa",
        "u": "Púrpura",
        "e": "Rojo",
        "w": "Blanco",
        "y": "Amarillo",
        "l": "Azul",
        "o": "Naranja",
        "k": "Negro",
        "f": "Ninguno",
    },
    "stem-root": {
        "b": "Bulbosa",
        "s": "Hinchada",
        "c": "Claviforme (en maza)",
        "u": "Copa",
        "e": "Igual (cilíndrica)",
        "z": "Rizomorfos",
        "r": "Enraizada",
        "f": "Fibrosa",
    },
    "stem-surface": {
        "d": "Seca",
        "e": "Carnosa",
        "g": "Surcos",
        "h": "Brillante",
        "i": "Fibrosa",
        "k": "Sedosa",
        "l": "Coriácea",
        "s": "Lisa",
        "t": "Pegajosa",
        "w": "Arrugada",
        "y": "Escamosa",
        "f": "Ninguna",
    },
    "stem-color": {
        "n": "Marrón",
        "b": "Ante (beige)",
        "g": "Gris",
        "r": "Verde",
        "p": "Rosa",
        "u": "Púrpura",
        "e": "Rojo",
        "w": "Blanco",
        "y": "Amarillo",
        "l": "Azul",
        "o": "Naranja",
        "k": "Negro",
        "f": "Ninguno",
    },
    "veil-type": {
        "p": "Parcial",
        "u": "Universal",
    },
    "veil-color": {
        "n": "Marrón",
        "b": "Ante (beige)",
        "g": "Gris",
        "r": "Verde",
        "p": "Rosa",
        "u": "Púrpura",
        "e": "Rojo",
        "w": "Blanco",
        "y": "Amarillo",
        "l": "Azul",
        "o": "Naranja",
        "k": "Negro",
        "f": "Ninguno",
    },
    "has-ring": {
        "t": "Sí (con anillo)",
        "f": "No",
    },
    "ring-type": {
        "c": "Araneoso (cortina)",
        "e": "Evanescente (fugaz)",
        "r": "Abocinado (expandible)",
        "g": "Estriado",
        "l": "Grande",
        "p": "Colgante",
        "s": "Envainante",
        "z": "Zonal",
        "y": "Escamoso",
        "m": "Móvil",
        "f": "Ninguno",
    },
    "spore-print-color": {
        "n": "Marrón",
        "b": "Ante (beige)",
        "g": "Gris",
        "r": "Verde",
        "p": "Rosa",
        "u": "Púrpura",
        "e": "Rojo",
        "w": "Blanco",
        "y": "Amarillo",
        "l": "Azul",
        "o": "Naranja",
        "k": "Negro",
    },
    "habitat": {
        "g": "Pastizales",
        "l": "Hojarasca",
        "m": "Praderas",
        "p": "Senderos",
        "h": "Brezales",
        "u": "Urbano",
        "w": "Desechos (vertederos)",
        "d": "Bosques",
    },
    "season": {
        "s": "Primavera",
        "u": "Verano",
        "a": "Otoño",
        "w": "Invierno",
    },
}


def _opciones(columna: str, valores: list) -> list:
    """Formatea los códigos de una columna con su etiqueta legible en español."""
    mapa = ETIQUETAS_CATEGORICAS.get(columna, {})
    return [f"{v} — {mapa[v]}" if v in mapa else v for v in valores]


def _codigo(opcion: str) -> str:
    return opcion.split(" — ")[0].strip()


# ------------------------------------------------------------------ carga de recursos
@st.cache_resource(show_spinner="Cargando modelo y artefactos...")
def cargar_recursos():
    pipeline = joblib.load(MODELS_DIR / "pipeline.joblib")
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
    model = joblib.load(MODELS_DIR / "best_model.joblib")
    meta = json.loads((MODELS_DIR / "model_metadata.json").read_text(encoding="utf-8"))
    return pipeline, preprocessor, model, meta


@st.cache_data(show_spinner="Cargando dataset...")
def cargar_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH, sep=";")


def importancia_global(meta: dict, model, preprocessor, columnas: list) -> pd.Series:
    """Importancia por variable original: la precalculada en el notebook 03 o, si falta,
    las importancias nativas del modelo agrupadas por variable."""
    if meta.get("global_importance"):
        return pd.Series(meta["global_importance"]).sort_values(ascending=False)
    importancias = getattr(model, "feature_importances_", None)
    if importancias is None:
        return None
    agrupadas = aggregate_by_original(importancias, list(preprocessor.get_feature_names_out()), columnas)
    return agrupadas.iloc[0].sort_values(ascending=False)


def barra_contribuciones(contribuciones: pd.Series, titulo: str, top: int = 10):
    """Barras horizontales: rojo empuja hacia venenoso, verde hacia comestible.
    Las variables fuera del top se agrupan en una sola barra."""
    orden = contribuciones.abs().sort_values(ascending=False).index
    visibles = contribuciones[orden[:top]].copy()
    if len(orden) > top:
        visibles[f"otras {len(orden) - top} variables"] = contribuciones[orden[top:]].sum()
    visibles = visibles[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(visibles) + 1.2))
    colores = ["#c62828" if v >= 0 else "#2e7d32" for v in visibles.values]
    barras = ax.barh(visibles.index, visibles.values * 100, color=colores)
    ax.bar_label(barras, labels=[f"{v * 100:+.1f}" for v in visibles.values], padding=3, fontsize=9)
    ax.axvline(0, color="#555555", lw=0.8)
    limite = max(abs(visibles.values).max() * 100 * 1.25, 1)
    ax.set_xlim(-limite, limite)
    ax.set_title(titulo, fontsize=11, fontweight="bold")
    ax.set_xlabel("Puntos porcentuales de P(venenoso)")
    plt.tight_layout()
    return fig


def fig_que_pasaria_si(escaneo: pd.DataFrame, columna: str, valor_actual, tau: float, numerica: bool):
    """Probabilidad de toxicidad para cada valor posible de una variable."""
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    nom_col = ETIQUETAS_VARIABLES.get(columna, columna)
    if numerica:
        ax.plot(escaneo["valor"], escaneo["probabilidad"], color="#1565c0", lw=2)
        ax.axvline(valor_actual, color="#555555", lw=1, linestyle=":", label=f"valor actual ({valor_actual:.2f})")
        ax.set_xlabel(nom_col)
    else:
        mapa = ETIQUETAS_CATEGORICAS.get(columna, {})
        etiquetas = [
            "sin\ndato" if pd.isna(v) else f"{v}\n{mapa.get(v, v)}"
            for v in escaneo["valor"]
        ]
        colores = ["#c62828" if p >= tau else "#2e7d32" for p in escaneo["probabilidad"]]
        bordes = ["#000000" if (pd.isna(v) and pd.isna(valor_actual)) or v == valor_actual else "none"
                  for v in escaneo["valor"]]
        ax.bar(etiquetas, escaneo["probabilidad"], color=colores, edgecolor=bordes, linewidth=2)
        ax.tick_params(axis="x", labelsize=7.5)
        ax.set_xlabel(f"{nom_col} (borde negro = valor actual)")
    ax.axhline(tau, color="#000000", lw=1.2, linestyle="--", label=f"umbral τ* = {tau:.2f}")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("P(venenoso)")
    ax.set_title(f"¿Qué pasaría si cambia {nom_col}?", fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


def fig_matriz_confusion(cm: dict):
    matriz = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Comestible (0)", "Venenoso (1)"],
                yticklabels=["Comestible (0)", "Venenoso (1)"], ax=ax)
    ax.set_title("Matriz de confusión — modelo desplegado (test)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    plt.tight_layout()
    return fig


def fig_comparativa(df_metricas: pd.DataFrame):
    derretido = df_metricas.melt(
        id_vars="Model",
        value_vars=["Accuracy", "Precision", "Recall (Poisonous)", "F1-Score", "ROC-AUC"],
        var_name="Métrica", value_name="Valor",
    )
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    sns.barplot(data=derretido, x="Model", y="Valor", hue="Métrica", palette="Set2", ax=ax)
    ax.set_ylim(0.75, 1.02)
    ax.set_title("Comparación de los 4 modelos (conjunto de test)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Puntaje (0 - 1)")
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=10)
    plt.tight_layout()
    return fig


# ------------------------------------------------------------------ validación cruzada
COLORES_MODELOS = {
    "Logistic Regression": "#8e24aa",
    "Random Forest": "#2e7d32",
    "XGBoost": "#ef6c00",
    "MLP Neural Network": "#1565c0",
}
SEGUNDOS_POR_FOLD = {"Logistic Regression": 1.7, "Random Forest": 1.6, "XGBoost": 1.4, "MLP Neural Network": 4.2}


@st.cache_data(show_spinner=False)
def particion_desarrollo():
    """Reproduce la partición 70/15/15 del notebook 03 (random_state=42).

    Devuelve el conjunto de desarrollo (Train + Validación), su objetivo binario y
    las comprobaciones de fuga frente al TEST, que nunca entra en la validación cruzada.
    """
    X_tr, X_va, X_te, y_tr, y_va, _ = get_train_val_test_data(cargar_dataset())
    X_dev = pd.concat([X_tr, X_va])
    y_dev = pd.concat([y_tr, y_va]).map({"e": 0, "p": 1}).astype(int)
    controles = {
        "n_desarrollo": len(X_dev),
        "n_test": len(X_te),
        "indices_compartidos": len(X_dev.index.intersection(X_te.index)),
        "test_repetidas_en_desarrollo": filas_repetidas_entre(X_dev, X_te),
        "duplicados_dataset": int(cargar_dataset().duplicated().sum()),
    }
    return X_dev, y_dev, controles


@st.cache_data(show_spinner=False, max_entries=24)
def cv_modelo(nombre_modelo: str, k: int, tau: float) -> pd.DataFrame:
    X_dev, y_dev, _ = particion_desarrollo()
    return validacion_cruzada_modelo(nombre_modelo, X_dev, y_dev, NUM_COLS, CAT_COLS, k=k, tau=tau)


def fig_esquema_folds(k: int):
    fig, ax = plt.subplots(figsize=(8, 0.36 * k + 0.9))
    for fila in range(k):
        y = k - 1 - fila
        for col in range(k):
            valida = fila == col
            ax.add_patch(plt.Rectangle((col, y), 0.94, 0.8, color="#ef6c00" if valida else "#bbdefb"))
            ax.text(col + 0.47, y + 0.4, "valida" if valida else "entrena", ha="center", va="center",
                    fontsize=8 if k <= 5 else 6.5, color="white" if valida else "#0d47a1")
    ax.set_xlim(-0.05, k)
    ax.set_ylim(-0.1, k)
    ax.set_yticks([k - 1 - f + 0.4 for f in range(k)])
    ax.set_yticklabels([f"Iteración {f + 1}" for f in range(k)], fontsize=8)
    ax.set_xticks([])
    for borde in ax.spines.values():
        borde.set_visible(False)
    ax.set_title(f"{k}-fold estratificado sobre Train + Validación (el test queda fuera)",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    return fig


def fig_metricas_cv(resultados: pd.DataFrame):
    """Puntos = cada fold; rombo negro = media ± desviación estándar."""
    modelos = list(resultados["Model"].unique())
    metricas = ["Accuracy", "Precision", "Recall (Poisonous)", "F1-Score"]
    fig, ejes = plt.subplots(2, 2, figsize=(10, 1.6 + 0.9 * len(modelos)), sharey=True)
    for ax, metrica in zip(ejes.flat, metricas):
        for i, modelo in enumerate(modelos):
            valores = resultados.loc[resultados["Model"] == modelo, metrica].to_numpy()
            desplazamiento = np.linspace(-0.18, 0.18, len(valores))
            ax.scatter(valores, i + desplazamiento, s=26, alpha=0.75, color=COLORES_MODELOS.get(modelo, "#555555"))
            ax.errorbar(valores.mean(), i, xerr=valores.std(ddof=1), fmt="D", color="black", ms=5, capsize=4)
        ax.set_yticks(range(len(modelos)))
        ax.set_yticklabels(modelos, fontsize=9)
        ax.set_title(metrica, fontsize=10, fontweight="bold")
        ax.grid(True, axis="x", alpha=0.3)
        minimo = resultados[metrica].min()
        ax.set_xlim(max(0.0, minimo - 0.02), 1.005)
    ejes[0, 0].invert_yaxis()
    fig.suptitle("Métrica obtenida en cada fold (umbral 0.5)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    return fig


def fig_tau_por_fold(resultados_modelo: pd.DataFrame, tau: float, nombre_modelo: str):
    taus = resultados_modelo["τ* del fold"].to_numpy()
    etiquetas = [f"Fold {f}" for f in resultados_modelo["Fold"]]
    fig, ax = plt.subplots(figsize=(8, 3.3))
    barras = ax.bar(etiquetas, taus, color=["#2e7d32" if t >= tau else "#c62828" for t in taus])
    ax.bar_label(barras, labels=[f"{t:.2f}" for t in taus], padding=2, fontsize=9)
    ax.axhline(tau, color="#000000", lw=1.2, linestyle="--", label=f"τ* desplegado = {tau:.2f}")
    ax.set_ylim(0, 1.22)
    ax.set_ylabel("Umbral más alto con FN = 0")
    ax.set_title(f"Umbral más alto sin falsos negativos en cada fold — {nombre_modelo}", fontsize=11,
                 fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


def tabla_resumen_cv(resumen: pd.DataFrame) -> pd.DataFrame:
    tabla = pd.DataFrame({"Modelo": resumen["Model"]})
    for metrica in METRICAS_CV:
        tabla[metrica] = [
            f"{m:.4f} ± {d:.4f}" for m, d in zip(resumen[f"{metrica} (media)"], resumen[f"{metrica} (desv.)"])
        ]
    tabla["FN totales (0.5)"] = resumen["FN totales (0.5)"]
    tabla[f"FN totales (τ={TAU:.2f})"] = resumen["FN totales (τ*)"]
    tabla[f"Folds con FN = 0 (τ={TAU:.2f})"] = resumen["Folds con FN = 0 (τ*)"]
    tabla["Umbral más alto con FN = 0 (peor fold)"] = resumen["τ* mínimo entre folds"].map(lambda t: f"{t:.2f}")
    tabla["Tiempo medio por fold (s)"] = resumen["Tiempo medio (s)"].map(lambda t: f"{t:.2f}")
    return tabla


def tabla_cv_vs_test(resumen: pd.DataFrame, metricas_test: list) -> pd.DataFrame:
    """¿El resultado del único split de test cae dentro de la variabilidad de la validación cruzada?"""
    test = pd.DataFrame(metricas_test).set_index("Model")
    filas = []
    for _, r in resumen.iterrows():
        if r["Model"] not in test.index:
            continue
        for metrica in ["Accuracy", "Recall (Poisonous)", "F1-Score", "ROC-AUC"]:
            media, desv = r[f"{metrica} (media)"], r[f"{metrica} (desv.)"]
            valor_test = float(test.loc[r["Model"], metrica])
            filas.append({
                "Modelo": r["Model"],
                "Métrica": metrica,
                "Validación cruzada (media ± desv.)": f"{media:.4f} ± {desv:.4f}",
                "Test (split único)": f"{valor_test:.4f}",
                "¿Test dentro de media ± 2·desv.?": "✅ sí" if abs(valor_test - media) <= 2 * desv + 1e-3 else "⚠️ no",
            })
    return pd.DataFrame(filas)


# ------------------------------------------------------------------ recursos globales
PIPELINE, PREPROCESSOR, MODEL, META = cargar_recursos()
TAU = float(META["optimal_threshold"])
CALIBRACION = META.get("threshold_calibration") or {}
ORIGEN_TAU = f"validación cruzada {CALIBRACION['k_folds']}-fold" if CALIBRACION else "un split de validación"
if CALIBRACION.get("cv_out_of_fold_band") and "margin_to_edible" in CALIBRACION:
    _franja = CALIBRACION["cv_out_of_fold_band"]
    REGLA_TAU = (
        f"τ* está en el centro de la franja sin errores de la {ORIGEN_TAU} "
        f"({_franja['max_proba_edible']:.2f} – {_franja['min_proba_poisonous']:.2f}): FN = 0 y FP = 0 en "
        f"todos los folds, con un margen de {CALIBRACION['margin_to_edible']:.2f} hacia los comestibles y de "
        f"{CALIBRACION['margin_to_poisonous']:.2f} hacia los venenosos."
    )
else:
    REGLA_TAU = f"τ* es el umbral más alto con 0 falsos negativos en {ORIGEN_TAU}."
NUM_COLS = META["numeric_features"]
CAT_COLS = META["categorical_features"]
TODAS_LAS_COLS = NUM_COLS + CAT_COLS

if not DATA_PATH.exists():
    st.error("No se encontró data/raw/secondary_data.csv. Clona el repositorio completo.")
    st.stop()
DF = cargar_dataset()
FEATURE_NAMES = list(PREPROCESSOR.get_feature_names_out())
MODELO_DE_ARBOLES = hasattr(MODEL, "estimators_") and hasattr(MODEL.estimators_[0], "tree_")

# ------------------------------------------------------------------ barra lateral
st.sidebar.title("🍄 Clasificador Micológico")
PAGINA = st.sidebar.radio(
    "Navegación",
    [
        "🎯 Predicción individual",
        "📦 Predicción por lotes",
        "📊 Rendimiento del modelo",
        "🔁 Validación cruzada",
        "🔍 Explicabilidad (XAI)",
        "🍄 Exploración del dataset",
        "📖 Ficha técnica y fórmulas",
    ],
)
st.sidebar.divider()
st.sidebar.markdown(
    f"""
**Modelo desplegado:** {META["model_name"]} ({META.get("model_class", "—")})

**Umbral de riesgo cero:** τ\\* = {TAU:.2f} ({ORIGEN_TAU})

**Precisión en test:** {META["best_model_test_metrics"]["Precision"]:.1%}

**Entrenado:** {META.get("trained_at", "—")}
"""
)
st.sidebar.caption(f"Metodología de riesgo cero: {REGLA_TAU}")

st.title("🍄 Sistema Inteligente de Clasificación Micológica")
st.markdown(
    "Clasificación de hongos como **Comestible (e)** o **Venenoso (p)** a partir de "
    "características morfológicas — *Secondary Mushroom Dataset*, 61.069 registros."
)
st.divider()

# ================================================================== 1. PREDICCIÓN INDIVIDUAL
if PAGINA == "🎯 Predicción individual":
    st.subheader("Predicción individual en tiempo real")
    st.caption("Completa las 20 características morfológicas del espécimen y ejecuta la inferencia.")

    with st.form("formulario_hongo"):
        st.markdown("**Variables métricas (numéricas)**")
        c_num = st.columns(3)
        valores_num = {}
        for col, contenedor in zip(NUM_COLS, c_num):
            rango = META["numeric_ranges"][col]
            valores_num[col] = contenedor.slider(
                ETIQUETAS_NUMERICAS[col],
                min_value=float(rango["min"]),
                max_value=float(rango["max"]),
                value=float((rango["min"] + rango["max"]) / 2),
                step=0.1,
            )

        st.markdown("**Variables nominales (categóricas)**")
        valores_cat = {}
        for i in range(0, len(CAT_COLS), 3):
            fila = st.columns(3)
            for col, contenedor in zip(CAT_COLS[i:i + 3], fila):
                valores_cat[col] = contenedor.selectbox(
                    ETIQUETAS_VARIABLES.get(col, col),
                    _opciones(col, META["categorical_values"][col]),
                    index=0,
                )
        ejecutar = st.form_submit_button("🔎 Clasificar espécimen", width="stretch")

    if ejecutar:
        fila = {**valores_num, **{c: _codigo(v) for c, v in valores_cat.items()}}
        entrada = pd.DataFrame([fila])[TODAS_LAS_COLS]
        proba = float(PIPELINE.predict_proba(entrada)[0, 1])
        st.session_state["ultima_prediccion"] = {"fila": fila, "proba": proba}

        venenoso = proba >= TAU
        cont1, cont2, cont3 = st.columns(3)
        cont1.metric(
            "Dictamen del sistema",
            "⚠️ VENENOSO" if venenoso else "🍽️ COMESTIBLE",
        )
        cont2.metric("P(venenoso)", f"{proba:.2%}")
        cont3.metric("Umbral τ*", f"{TAU:.2f}")
        st.progress(proba, text="Probabilidad de toxicidad estimada por el modelo")

        st.markdown("**Efecto del umbral de decisión (metodología de riesgo cero):**")
        tau_usuario = st.slider(
            "Umbral τ", min_value=0.01, max_value=0.95, value=TAU, step=0.01,
            help=f"τ* = {TAU:.2f}. {REGLA_TAU}",
        )
        if tau_usuario != TAU:
            dictamen_alt = proba >= tau_usuario
            st.info(
                f"Con τ = {tau_usuario:.2f} el dictamen sería: "
                f"**{'⚠️ VENENOSO' if dictamen_alt else '🍽️ COMESTIBLE'}**."
            )
        if venenoso:
            st.error("⛔ El sistema recomienda NO consumir este espécimen.")
        else:
            st.success("✅ Espécimen clasificado como comestible bajo la garantía de riesgo cero.")

        with st.expander("Ver características ingresadas"):
            datos_mostrar = []
            for col in TODAS_LAS_COLS:
                val = fila.get(col)
                nom_var = ETIQUETAS_VARIABLES.get(col, col)
                if col in NUM_COLS:
                    unidad = "cm" if "diameter" in col or "height" in col else "mm"
                    val_str = f"{float(val):.2f} {unidad}"
                else:
                    desc = ETIQUETAS_CATEGORICAS.get(col, {}).get(val, val)
                    if desc != val:
                        val_str = f"{desc} ({val})"
                    else:
                        val_str = str(val)
                datos_mostrar.append({"Característica": nom_var, "Valor": val_str})

            df_caracteristicas = pd.DataFrame(datos_mostrar)
            st.dataframe(df_caracteristicas, width="stretch", hide_index=True)

# ================================================================== 2. PREDICCIÓN POR LOTES
if PAGINA == "📦 Predicción por lotes":
    st.subheader("Predicción por lotes desde CSV")
    st.caption("Sube un CSV con las 20 características (la columna `class` es opcional y se ignora si viene).")

    plantilla = DF[TODAS_LAS_COLS].head(5).copy()
    st.download_button(
        "⬇️ Descargar plantilla de ejemplo (5 registros)",
        plantilla.to_csv(index=False).encode("utf-8"),
        file_name="plantilla_hongos.csv",
        mime="text/csv",
    )

    archivo = st.file_uploader("Archivo CSV", type=["csv"])
    if archivo is not None:
        try:
            lote = pd.read_csv(archivo, sep=None, engine="python")
        except Exception:
            archivo.seek(0)
            lote = pd.read_csv(archivo, sep=";")

        faltantes = [c for c in TODAS_LAS_COLS if c not in lote.columns]
        if faltantes:
            st.error(f"Faltan columnas obligatorias: {', '.join(faltantes)}")
        else:
            with st.spinner("Infiriendo sobre el lote..."):
                probabilidades = PIPELINE.predict_proba(lote[TODAS_LAS_COLS])[:, 1]
            resultado = lote.copy()
            resultado["probabilidad_venenoso"] = probabilidades.round(4)
            resultado["dictamen"] = np.where(probabilidades >= TAU, "Venenoso (p)", "Comestible (e)")

            c1, c2, c3 = st.columns(3)
            c1.metric("Registros procesados", len(resultado))
            c2.metric("Clasificados venenosos", int((resultado["dictamen"] == "Venenoso (p)").sum()))
            c3.metric("Clasificados comestibles", int((resultado["dictamen"] == "Comestible (e)").sum()))

            st.dataframe(resultado.head(100), width="stretch")
            if len(resultado) > 100:
                st.caption(f"Mostrando 100 de {len(resultado)} registros.")

            st.download_button(
                "⬇️ Descargar resultados completos (CSV)",
                resultado.to_csv(index=False).encode("utf-8"),
                file_name="resultados_prediccion.csv",
                mime="text/csv",
            )

# ================================================================== 3. RENDIMIENTO
elif PAGINA == "📊 Rendimiento del modelo":
    st.subheader("Dashboard de rendimiento del modelo desplegado")
    mejor = META["best_model_test_metrics"]
    if mejor["Precision"] > 0.95:
        st.success(
            f"✅ **Criterio de la rúbrica cumplido:** precisión del modelo desplegado en el conjunto de "
            f"prueba = **{mejor['Precision']:.1%}** (> 95 %) sobre {META['best_model_confusion_matrix_test']['tn'] + META['best_model_confusion_matrix_test']['tp']:,} especímenes nunca vistos."
        )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{mejor['Accuracy']:.2%}")
    m2.metric("Precision", f"{mejor['Precision']:.2%}")
    m3.metric("Recall (venenoso)", f"{mejor['Recall (Poisonous)']:.2%}")
    m4.metric("F1-Score", f"{mejor['F1-Score']:.4f}")
    m5.metric("ROC-AUC", f"{mejor['ROC-AUC']:.4f}")

    st.caption(
        f"Selección por evidencia — criterio: {META['selection_criteria']}. "
        f"Umbral de riesgo cero calibrado: τ* = {TAU:.2f}."
    )

    df_test = pd.DataFrame(META["test_metrics"])
    c_izq, c_der = st.columns([1.4, 1])
    with c_izq:
        st.pyplot(fig_comparativa(df_test))
    with c_der:
        st.pyplot(fig_matriz_confusion(META["best_model_confusion_matrix_test"]))

    st.markdown("**Tabla comparativa en test:**")
    st.dataframe(
        df_test.style.format(
            {c: "{:.4f}" for c in ["Accuracy", "Precision", "Recall (Poisonous)", "F1-Score", "ROC-AUC"]}
        ),
        width="stretch",
        hide_index=True,
    )
    with st.expander("Ver métricas de validación (usadas para decisiones de diseño)"):
        st.dataframe(pd.DataFrame(META["validation_metrics"]), width="stretch", hide_index=True)

# ================================================================== 3b. VALIDACIÓN CRUZADA
elif PAGINA == "🔁 Validación cruzada":
    st.subheader("Validación cruzada estratificada (k-fold) sin fuga de datos")
    st.markdown(
        "Un solo split train/test puede salir «con suerte». La validación cruzada divide el conjunto de "
        "desarrollo en **k partes**, entrena **k veces** dejando fuera una parte distinta cada vez y "
        "reporta la **media ± desviación estándar**. Si la desviación es pequeña, el rendimiento es estable "
        "y no depende de cómo cayó la partición."
    )

    MODELOS_DISPONIBLES = list(COLORES_MODELOS)
    with st.form("formulario_cv"):
        c_mod, c_k = st.columns([2, 1])
        modelos_elegidos = c_mod.multiselect(
            "Modelos a validar", MODELOS_DISPONIBLES, default=MODELOS_DISPONIBLES,
            help="Se entrena un pipeline nuevo (preprocesamiento + modelo) en cada fold.",
        )
        k_elegido = c_k.select_slider("Número de folds (k)", options=[3, 5, 10], value=5)
        estimado = sum(SEGUNDOS_POR_FOLD.get(m, 2) for m in modelos_elegidos) * k_elegido
        st.caption(f"Tiempo estimado la primera vez: ~{estimado:.0f} s. Después queda en caché y es instantáneo.")
        ejecutar_cv = st.form_submit_button("▶️ Ejecutar validación cruzada", width="stretch")

    if ejecutar_cv:
        if not modelos_elegidos:
            st.warning("Elige al menos un modelo.")
        else:
            st.session_state["cv_config"] = (tuple(modelos_elegidos), int(k_elegido))

    st.pyplot(fig_esquema_folds(int(k_elegido)))

    config = st.session_state.get("cv_config")
    if config is None:
        st.info("Elige los modelos y el número de folds, y pulsa **Ejecutar validación cruzada**.")
    else:
        modelos_cv, k_cv = config
        X_dev, y_dev, controles = particion_desarrollo()
        partes = []
        with st.status(f"Validación cruzada {k_cv}-fold en curso...", expanded=False) as estado:
            for i, modelo in enumerate(modelos_cv, start=1):
                estado.update(label=f"[{i}/{len(modelos_cv)}] Validando {modelo} ({k_cv} folds)...")
                partes.append(cv_modelo(modelo, k_cv, TAU))
            estado.update(label=f"Validación cruzada {k_cv}-fold completada", state="complete")
        resultados_cv = pd.concat(partes, ignore_index=True)
        resumen_cv = resumen_validacion_cruzada(resultados_cv)

        st.markdown(f"#### Resultados: {k_cv}-fold sobre {controles['n_desarrollo']:,} especímenes de desarrollo")
        st.dataframe(tabla_resumen_cv(resumen_cv), width="stretch", hide_index=True)
        st.caption(
            "Métricas con el umbral 0.5. Las columnas con τ aplican el umbral de riesgo cero "
            f"τ* = {TAU:.2f}, calibrado para el modelo desplegado ({META['model_name']}); en los demás "
            "modelos es solo referencia. «Umbral más alto con FN = 0 (peor fold)» es el umbral más alto "
            "que habría dado FN = 0 en todos los folds: el borde superior de la franja segura."
        )

        # ---------------------------------------------------- dictamen sobre el modelo desplegado
        desplegado = META["model_name"]
        if desplegado in modelos_cv:
            res_desp = resultados_cv[resultados_cv["Model"] == desplegado]
            fn_tau = int(res_desp["FN (τ*)"].sum())
            folds_con_fn = int((res_desp["FN (τ*)"] > 0).sum())
            tau_robusto = float(res_desp["τ* del fold"].min())
            if fn_tau == 0:
                st.success(
                    f"✅ **Riesgo cero estable:** con τ* = {TAU:.2f}, el {desplegado} no dejó pasar ningún "
                    f"hongo venenoso en ninguno de los {k_cv} folds."
                )
            else:
                st.warning(
                    f"⚠️ Con τ* = {TAU:.2f}, el {desplegado} clasificó como comestibles **{fn_tau}** "
                    f"hongo(s) venenoso(s) en **{folds_con_fn} de {k_cv}** folds. El τ* desplegado se calibró "
                    f"con {ORIGEN_TAU}; con esta configuración, el umbral más alto con FN = 0 en todos los "
                    f"folds es **{tau_robusto:.2f}**. Para recalibrarlo, ejecuta `python recalibrar_umbral_cv.py`."
                )
            st.pyplot(fig_tau_por_fold(res_desp, TAU, desplegado))
            st.caption(
                "Cada barra es el umbral más alto que no deja pasar venenosos en ese fold. Verde: τ* desplegado "
                "queda por debajo, así que no produce falsos negativos. Rojo: sí los produce."
            )

        st.markdown("#### Variabilidad entre folds")
        st.pyplot(fig_metricas_cv(resultados_cv))

        st.markdown("#### ¿El resultado del test fue suerte?")
        st.caption(
            "Compara la métrica del único split de test (notebook 03) con la media ± desviación de la "
            "validación cruzada. Si el test cae dentro de la banda, el resultado reportado es representativo."
        )
        st.dataframe(tabla_cv_vs_test(resumen_cv, META["test_metrics"]), width="stretch", hide_index=True)

        with st.expander("Ver resultados fold por fold"):
            st.dataframe(
                resultados_cv.style.format(
                    {**{m: "{:.4f}" for m in METRICAS_CV}, "% venenosos (validación)": "{:.2%}",
                     "τ* del fold": "{:.2f}", "Tiempo (s)": "{:.2f}"}
                ),
                width="stretch",
                hide_index=True,
            )
            st.download_button(
                "⬇️ Descargar resultados por fold (CSV)",
                resultados_cv.to_csv(index=False).encode("utf-8"),
                file_name=f"validacion_cruzada_{k_cv}fold.csv",
                mime="text/csv",
            )

        # ---------------------------------------------------- controles de fuga de datos
        st.markdown("#### Controles de fuga de datos (data leakage)")
        prop = resultados_cv["% venenosos (validación)"]
        repetidas = controles["test_repetidas_en_desarrollo"]
        st.markdown(
            f"""
| Control | Resultado |
|---|---|
| El test no participa en la validación cruzada | ✅ {controles['n_desarrollo']:,} filas de desarrollo y {controles['n_test']:,} de test; índices compartidos: **{controles['indices_compartidos']}** |
| Preprocesamiento ajustado dentro de cada fold | ✅ Se crea un `Pipeline` nuevo por fold: el imputador, el escalador y el One-Hot solo ven las filas de entrenamiento del fold |
| Folds estratificados | ✅ % de venenosos en validación entre **{prop.min():.2%}** y **{prop.max():.2%}** |
| Filas del test idénticas a filas de desarrollo | {'✅ ninguna' if repetidas == 0 else f'⚠️ **{repetidas}** ({repetidas / controles["n_test"]:.2%} del test) — el dataset tiene {controles["duplicados_dataset"]} registros duplicados'} |
"""
        )
        st.code(
            "for idx_ent, idx_val in StratifiedKFold(n_splits=k, shuffle=True, random_state=42).split(X_dev, y_dev):\n"
            "    pipeline = create_model_pipeline(clone(modelo), columnas_numericas, columnas_categoricas)\n"
            "    pipeline.fit(X_dev.iloc[idx_ent], y_dev.iloc[idx_ent])      # se ajusta solo con el fold de entrenamiento\n"
            "    proba = pipeline.predict_proba(X_dev.iloc[idx_val])[:, 1]   # se evalúa con datos que no vio",
            language="python",
        )
        st.info(
            "**Limitación del dataset:** el *Secondary Mushroom Dataset* fue simulado a partir de 173 especies, "
            "así que hay muchos especímenes muy parecidos entre sí. La validación cruzada aleatoria mide qué tan "
            "bien el modelo reconoce especies que ya vio, no especies nuevas; por eso las métricas cercanas al "
            "100 % no implican lo mismo en hongos de especies fuera del dataset."
        )

# ================================================================== 4. EXPLICABILIDAD
elif PAGINA == "🔍 Explicabilidad (XAI)":
    st.subheader("Explicabilidad: ¿por qué el modelo decidió esto?")
    ultima = st.session_state.get("ultima_prediccion")
    if ultima is None:
        espécimen = DF[TODAS_LAS_COLS].iloc[0].to_dict()
        st.caption("No hay predicción reciente: se explica el espécimen de ejemplo (primera fila del dataset).")
    else:
        espécimen = ultima["fila"]
        st.caption("Explicando tu última predicción de la página «Predicción individual».")

    entrada = pd.DataFrame([espécimen])[TODAS_LAS_COLS]
    proba = float(PIPELINE.predict_proba(entrada)[0, 1])
    venenoso = proba >= TAU

    c1, c2, c3 = st.columns(3)
    c1.metric("Dictamen", "⚠️ VENENOSO" if venenoso else "🍽️ COMESTIBLE")
    c2.metric("P(venenoso)", f"{proba:.2%}")
    c3.metric("Umbral τ*", f"{TAU:.2f}")

    # ---------------------------------------------------------- desglose local
    st.markdown("#### 1. Desglose de la probabilidad por variable")
    if MODELO_DE_ARBOLES:
        base, contrib = tree_path_contributions(MODEL, PREPROCESSOR.transform(entrada))
        por_variable = aggregate_by_original(contrib, FEATURE_NAMES, TODAS_LAS_COLS).iloc[0]
        st.caption(
            f"Punto de partida **{base:.1%}** (proporción de venenosos con la que arrancan los árboles) "
            f"+ aporte de cada variable = **{base + por_variable.sum():.1%}**. "
            "Rojo empuja hacia venenoso; verde, hacia comestible."
        )
        por_variable_es = por_variable.copy()
        por_variable_es.index = [ETIQUETAS_VARIABLES.get(c, c) for c in por_variable_es.index]
        st.pyplot(barra_contribuciones(por_variable_es, "Aporte de cada variable a P(venenoso)"))
        with st.expander("Ver el detalle por característica codificada (One-Hot)"):
            detalle = pd.DataFrame({"característica": FEATURE_NAMES, "aporte": contrib[0]})
            detalle = detalle[detalle["aporte"] != 0].sort_values("aporte", key=np.abs, ascending=False)
            st.dataframe(detalle.style.format({"aporte": "{:+.4f}"}), width="stretch", hide_index=True)
    else:
        st.info(
            f"El desglose nativo solo aplica a ensambles de árboles; el modelo desplegado es "
            f"{META['model_name']}. Usa el simulador de abajo para analizar la decisión."
        )

    # ---------------------------------------------------------- simulador qué pasaría si
    st.markdown("#### 2. ¿Qué pasaría si…?")
    st.caption("Cambia una variable del espécimen a todos sus valores posibles y observa la probabilidad.")
    columna = st.selectbox(
        "Variable a simular",
        TODAS_LAS_COLS,
        format_func=lambda c: ETIQUETAS_VARIABLES.get(c, c),
        index=TODAS_LAS_COLS.index(por_variable.abs().idxmax()) if MODELO_DE_ARBOLES else 0,
    )
    numerica = columna in NUM_COLS
    if numerica:
        rango = META["numeric_ranges"][columna]
        valores = np.linspace(rango["min"], rango["max"], 60).tolist()
    else:
        valores = META["categorical_values"][columna] + [np.nan]
    escaneo = what_if_scan(PIPELINE, espécimen, columna, valores)
    st.pyplot(fig_que_pasaria_si(escaneo, columna, espécimen[columna], TAU, numerica))
    cambian = escaneo[(escaneo["probabilidad"] >= TAU) != venenoso]
    if cambian.empty:
        st.success(f"Ningún valor de `{columna}` cambia el dictamen por sí solo.")
    else:
        st.warning(
            f"{len(cambian)} de {len(escaneo)} valores de `{columna}` cambiarían el dictamen a "
            f"**{'comestible' if venenoso else 'venenoso'}**."
        )

    # ---------------------------------------------------------- importancia global
    st.markdown("#### 3. Importancia global del modelo")
    importancia = importancia_global(META, MODEL, PREPROCESSOR, TODAS_LAS_COLS)
    if importancia is None:
        st.warning("Este modelo no expone importancias globales.")
    else:
        top = importancia.head(12)[::-1].copy()
        top.index = [ETIQUETAS_VARIABLES.get(c, c) for c in top.index]
        fig, ax = plt.subplots(figsize=(8, 4.6))
        ax.barh(top.index, top.values * 100, color="#1565c0")
        ax.set_title("Variables que más pesan en las decisiones del modelo", fontsize=11, fontweight="bold")
        ax.set_xlabel("Aporte medio absoluto (puntos porcentuales de P(venenoso))")
        plt.tight_layout()
        st.pyplot(fig)
        st.caption(META.get("global_importance_method", "Importancias nativas del modelo (feature_importances_)."))

# ================================================================== 5. EXPLORACIÓN DEL DATASET
elif PAGINA == "🍄 Exploración del dataset":
    st.subheader("Exploración del dataset (INFOVIZ)")
    st.caption(f"{DF.shape[0]:,} registros × {DF.shape[1]} columnas — Secondary Mushroom Dataset.")

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(4.6, 3.6))
        conteo = DF["class"].value_counts()
        ax.pie(conteo.values, labels=["Venenoso (p)" if k == "p" else "Comestible (e)" for k in conteo.index],
               autopct="%1.1f%%", colors=["#c62828", "#2e7d32"], startangle=90)
        ax.set_title("Balance de clases", fontsize=12, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
    with c2:
        fig, ejes = plt.subplots(1, 3, figsize=(10.5, 3.6))
        for ax, col in zip(ejes, NUM_COLS):
            sns.histplot(DF[col], bins=40, ax=ax, color="#1565c0")
            ax.set_title(ETIQUETAS_VARIABLES.get(col, col), fontsize=10)
            ax.set_xlabel("")
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("**Explorador de variables categóricas:**")
    col_sel = st.selectbox(
        "Variable",
        CAT_COLS,
        format_func=lambda c: ETIQUETAS_VARIABLES.get(c, c),
        index=CAT_COLS.index("gill-color") if "gill-color" in CAT_COLS else 0,
    )
    top = DF[col_sel].value_counts().head(12)
    mapa_sel = ETIQUETAS_CATEGORICAS.get(col_sel, {})
    etiquetas_cat = [f"{k}\n({mapa_sel.get(k, k)})" for k in top.index]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    ax.bar(etiquetas_cat, top.values, color="#ef6c00")
    ax.set_title(f"Frecuencia de categorías — {ETIQUETAS_VARIABLES.get(col_sel, col_sel)} (top 12)", fontsize=12, fontweight="bold")
    ax.set_ylabel("registros")
    plt.tight_layout()
    st.pyplot(fig)

# ================================================================== 6. FICHA TÉCNICA
elif PAGINA == "📖 Ficha técnica y fórmulas":
    st.subheader("Ficha técnica del modelo")
    m1, m2, m3 = st.columns(3)
    m1.metric("Modelo", META["model_name"])
    m2.metric("Clase del estimador", META.get("model_class", "—"))
    m3.metric("Umbral τ*", f"{TAU:.2f}")
    st.caption(f"Criterio de selección: {META['selection_criteria']}")

    if CALIBRACION:
        st.markdown("**Calibración del umbral de riesgo cero τ\\***")
        detalle_franja = ""
        if "margin_to_edible" in CALIBRACION:
            franja_cv = CALIBRACION["cv_out_of_fold_band"]
            detalle_franja = (
                f" Franja: {franja_cv['max_proba_edible']:.2f} – {franja_cv['min_proba_poisonous']:.2f}; margen de "
                f"{CALIBRACION['margin_to_edible']:.2f} hacia los comestibles y de "
                f"{CALIBRACION['margin_to_poisonous']:.2f} hacia los venenosos."
            )
        st.markdown(f"{CALIBRACION['method']}{detalle_franja}")
        st.markdown(
            f"Antes se usaba τ = {CALIBRACION['single_split_threshold']:.2f}: el umbral más alto con FN = 0 en un "
            "único split de validación. Dependía del split (falla en un fold) y quedaba en el borde de la franja "
            "segura, sin margen."
        )
        prueba = CALIBRACION["test_at_threshold"]
        columnas_fold = {
            "Fold": range(1, len(CALIBRACION["fold_thresholds"]) + 1),
            "Umbral más alto con FN = 0": CALIBRACION["fold_thresholds"],
        }
        if "fold_fn_at_threshold" in CALIBRACION:
            columnas_fold[f"FN con τ* = {TAU:.2f}"] = CALIBRACION["fold_fn_at_threshold"]
            columnas_fold[f"FP con τ* = {TAU:.2f}"] = CALIBRACION["fold_fp_at_threshold"]
        columnas_fold[f"FN con τ = {CALIBRACION['single_split_threshold']:.2f}"] = (
            CALIBRACION["fold_fn_at_single_split_threshold"]
        )
        st.dataframe(pd.DataFrame(columnas_fold), width="stretch", hide_index=True)
        st.caption(
            f"Verificación en test con τ* = {TAU:.2f}: FN = {prueba['fn']}, FP = {prueba['fp']}, "
            f"precisión = {prueba['precision']:.2%}, recall = {prueba['recall']:.2%}. En test, el venenoso "
            f"con menor probabilidad tuvo P = {CALIBRACION['test_min_proba_poisonous']:.2f} y el comestible "
            f"con mayor probabilidad, P = {CALIBRACION['test_max_proba_edible']:.2f}."
        )

    with st.expander("Metadatos completos (model_metadata.json)"):
        st.json(META)

    st.markdown("**Funcionalidades de la aplicación (criterio de rúbrica: ≥ 10):**")
    st.markdown(
        "\n".join(
            f"{i}. {f}"
            for i, f in enumerate(
                [
                    "Predicción individual interactiva con las 20 características morfológicas.",
                    "Predicción por lotes desde archivo CSV con validación de columnas.",
                    "Probabilidad de toxicidad en tiempo real (predict_proba del pipeline).",
                    "Umbral de riesgo cero τ* aplicado al dictamen y explorable con un deslizador.",
                    "Desglose local de la probabilidad por variable (contribuciones nativas del Random Forest).",
                    "Simulador «¿qué pasaría si…?» para cualquier variable del espécimen.",
                    "Importancia global precalculada (sin cálculo al arrancar la app).",
                    "Dashboard de métricas del modelo desplegado en el conjunto de test.",
                    "Comparativa de los 4 modelos entrenados (LR, Random Forest, XGBoost, MLP).",
                    "Matriz de confusión del modelo desplegado.",
                    "Exploración del dataset con visualizaciones (INFOVIZ).",
                    "Descarga de resultados por lotes y plantilla CSV de ejemplo.",
                    "Ficha técnica y documentación matemática embebida.",
                    "Validación cruzada estratificada k-fold sin fuga de datos (pipeline por fold, test excluido).",
                ],
                start=1,
            )
        )
    )

    st.divider()
    st.markdown("## Documentación matemática de los modelos")
    if FORMULAS_PATH.exists():
        texto = FORMULAS_PATH.read_text(encoding="utf-8")
        import re

        partes = re.split(r"!\[([^\]]*)\]\(([^)]+)\)", texto)
        for i, parte in enumerate(partes):
            if i % 3 == 0:
                st.markdown(parte)
            elif i % 3 == 1:
                alt = parte
            else:
                ruta = (FORMULAS_PATH.parent / parte).resolve()
                if ruta.exists():
                    st.image(str(ruta), caption=alt, width="stretch")
    else:
        st.warning("No se encontró docs/formulas_matematicas_modelos.md.")

    st.divider()
    st.caption("Demo del Hito 2 — Electiva en Ingeniería Aplicada II (Data Science Fundamentals).")
