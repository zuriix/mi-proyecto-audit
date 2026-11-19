import os
from pathlib import Path
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.tree import DecisionTreeClassifier

# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

ARCHIVO_TRAIN = BASE_DIR / "data" / "processed" / "train.csv"
ARCHIVO_VALIDACION = BASE_DIR / "data" / "processed" / "validation.csv"

RANDOM_STATE = 42

CARPETA_RESULTADOS = BASE_DIR / "data" / "results"
os.makedirs(CARPETA_RESULTADOS, exist_ok=True)


# ============================================================
# INICIO
# ============================================================

print()
print("===================================")
print("ENTRENAMIENTO Y COMPARACIÓN DE MODELOS (ETAPA 3)")
print("EVALUACIÓN ESTRICTA EN VALIDACIÓN POR GRUPOS")
print("===================================")
print()


# ============================================================
# CARGAR DATOS
# ============================================================

train = pd.read_csv(ARCHIVO_TRAIN)
validation = pd.read_csv(ARCHIVO_VALIDACION)

print(f"Datos de entrenamiento : {len(train):>5} registros, {train['paquete'].nunique()} paquetes")
print(f"Datos de validación    : {len(validation):>5} registros, {validation['paquete'].nunique()} paquetes")
print()


# ============================================================
# CARACTERÍSTICAS DEL MODELO
# ============================================================
#
# NO incluimos:
# - paquete (identificador)
# - version (identificador)
# - numero_vulnerabilidades (fuga de información)
# - riesgo (variable objetivo)
# ============================================================

CARACTERISTICAS = [
    "edad_version_dias",
    "versiones_anteriores",
    "numero_dependencias",
    "numero_dependencias_opcionales",
    "numero_scripts",
    "tiene_repositorio",
    "tiene_licencia",
]

VARIABLE_OBJETIVO = "riesgo"


# ============================================================
# VERIFICAR COLUMNAS
# ============================================================

columnas_faltantes_train = [
    col for col in CARACTERISTICAS + [VARIABLE_OBJETIVO] if col not in train.columns
]
columnas_faltantes_val = [
    col for col in CARACTERISTICAS + [VARIABLE_OBJETIVO] if col not in validation.columns
]

if columnas_faltantes_train or columnas_faltantes_val:
    print(f"ERROR: Columnas faltantes en train: {columnas_faltantes_train}, en val: {columnas_faltantes_val}")
    raise SystemExit(1)


# ============================================================
# PREPARAR X E Y
# ============================================================

X_train = train[CARACTERISTICAS]
y_train = train[VARIABLE_OBJETIVO]

X_validation = validation[CARACTERISTICAS]
y_validation = validation[VARIABLE_OBJETIVO]

print("Características utilizadas:")
for c in CARACTERISTICAS:
    print(f" - {c}")
print(f"Variable objetivo: {VARIABLE_OBJETIVO}\n")


# ============================================================
# DEFINIR MODELOS (CONFIGURACIÓN ESTÁNDAR DEL PROYECTO)
# ============================================================

modelos = {
    "Regresion Logistica": LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
    ),
    "Arbol de Decision": DecisionTreeClassifier(
        max_depth=5,
        random_state=RANDOM_STATE,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    ),
}


# ============================================================
# ENTRENAMIENTO Y EVALUACIÓN EN VALIDACIÓN
# ============================================================

resultados = []

for nombre, modelo in modelos.items():
    print("===================================")
    print("MODELO:", nombre)
    print("===================================")

    modelo.fit(X_train, y_train)
    predicciones = modelo.predict(X_validation)

    acc = accuracy_score(y_validation, predicciones)
    prec = precision_score(y_validation, predicciones, zero_division=0)
    rec = recall_score(y_validation, predicciones, zero_division=0)
    f1 = f1_score(y_validation, predicciones, zero_division=0)

    matriz = confusion_matrix(y_validation, predicciones, labels=[0, 1])
    tn, fp, fn, tp = int(matriz[0, 0]), int(matriz[0, 1]), int(matriz[1, 0]), int(matriz[1, 1])

    print(f"Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision: {prec:.4f} ({prec*100:.2f}%)")
    print(f"Recall   : {rec:.4f} ({rec*100:.2f}%)")
    print(f"F1-score : {f1:.4f} ({f1*100:.2f}%)")
    print("Matriz de confusión:")
    print(f"  TN={tn:<5} FP={fp:<5}")
    print(f"  FN={fn:<5} TP={tp:<5}")
    print()

    resultados.append({
        "modelo": nombre,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    })


# ============================================================
# COMPARACIÓN Y SELECCIÓN BASADA EN VALIDACIÓN
# ============================================================

resultados_df = pd.DataFrame(resultados)

# Regla explícita de decisión del proyecto: orden descendente por F1-score
resultados_df = resultados_df.sort_values(by="f1_score", ascending=False)

print("===================================")
print("TABLA COMPARATIVA EN VALIDACIÓN")
print("===================================")
print(resultados_df[["modelo", "accuracy", "precision", "recall", "f1_score", "tn", "fp", "fn", "tp"]].to_string(index=False))
print()

mejor_modelo = resultados_df.iloc[0]

print("===================================")
print("MODELO SELECCIONADO POR VALIDACIÓN")
print("===================================")
print(f"Modelo      : {mejor_modelo['modelo']}")
print(f"F1-score    : {mejor_modelo['f1_score']:.4f}")
print(f"Recall      : {mejor_modelo['recall']:.4f}")
print(f"Precision   : {mejor_modelo['precision']:.4f}")
print(f"Accuracy    : {mejor_modelo['accuracy']:.4f}")
print(f"Justificación: Mayor equilibrio armónico entre Recall ({mejor_modelo['recall']:.2%}) "
      f"y Precisión ({mejor_modelo['precision']:.2%}) con F1-score de {mejor_modelo['f1_score']:.4f}.")
print()


# ============================================================
# GUARDAR RESULTADOS
# ============================================================

ARCHIVO_ETAPA3 = CARPETA_RESULTADOS / "comparacion_modelos_etapa3.csv"
ARCHIVO_OFICIAL = CARPETA_RESULTADOS / "comparacion_modelos.csv"

resultados_df.to_csv(ARCHIVO_ETAPA3, index=False, encoding="utf-8")
resultados_df.to_csv(ARCHIVO_OFICIAL, index=False, encoding="utf-8")

print(f"Resultados guardados en:")
print(f" - {ARCHIVO_ETAPA3}")
print(f" - {ARCHIVO_OFICIAL}")
print()
print("===================================")
print("PROCESO TERMINADO")
print("===================================")