import pandas as pd
import os

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

ARCHIVO_TRAIN = "data/processed/train.csv"
ARCHIVO_VALIDACION = "data/processed/validation.csv"

RANDOM_STATE = 42

CARPETA_RESULTADOS = "data/results"

os.makedirs(CARPETA_RESULTADOS, exist_ok=True)


# ============================================================
# INICIO
# ============================================================

print()
print("===================================")
print("ENTRENAMIENTO DE MODELOS")
print("===================================")
print()


# ============================================================
# CARGAR DATOS
# ============================================================

train = pd.read_csv(ARCHIVO_TRAIN)
validation = pd.read_csv(ARCHIVO_VALIDACION)

print("Datos de entrenamiento:", len(train))
print("Datos de validación:", len(validation))
print()


# ============================================================
# CARACTERÍSTICAS DEL MODELO
# ============================================================
#
# IMPORTANTE:
#
# NO incluimos:
# - paquete
# - version
# - numero_vulnerabilidades
#
# "numero_vulnerabilidades" NO puede ser una característica
# porque se utiliza para determinar la variable objetivo "riesgo".
# Incluirla produciría fuga de información.
#
# ============================================================

CARACTERISTICAS = [
    "edad_version_dias",
    "versiones_anteriores",
    "numero_dependencias",
    "numero_dependencias_opcionales",
    "numero_scripts",
    "tiene_repositorio",
    "tiene_licencia"
]

VARIABLE_OBJETIVO = "riesgo"


# ============================================================
# VERIFICAR COLUMNAS
# ============================================================

columnas_faltantes_train = [
    columna
    for columna in CARACTERISTICAS + [VARIABLE_OBJETIVO]
    if columna not in train.columns
]

columnas_faltantes_validation = [
    columna
    for columna in CARACTERISTICAS + [VARIABLE_OBJETIVO]
    if columna not in validation.columns
]

if columnas_faltantes_train:
    print("ERROR: faltan columnas en train.csv:")
    print(columnas_faltantes_train)
    raise SystemExit

if columnas_faltantes_validation:
    print("ERROR: faltan columnas en validation.csv:")
    print(columnas_faltantes_validation)
    raise SystemExit


# ============================================================
# PREPARAR X E Y
# ============================================================

X_train = train[CARACTERISTICAS]
y_train = train[VARIABLE_OBJETIVO]

X_validation = validation[CARACTERISTICAS]
y_validation = validation[VARIABLE_OBJETIVO]


print("Características utilizadas:")
for caracteristica in CARACTERISTICAS:
    print("-", caracteristica)

print()

print("Variable objetivo:", VARIABLE_OBJETIVO)
print()


# ============================================================
# DEFINIR MODELOS
# ============================================================

modelos = {
    "Regresion Logistica": LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE
    ),

    "Arbol de Decision": DecisionTreeClassifier(
        max_depth=5,
        random_state=RANDOM_STATE
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=RANDOM_STATE,
        class_weight="balanced"
    )
}


# ============================================================
# ENTRENAMIENTO Y EVALUACIÓN
# ============================================================

resultados = []


for nombre, modelo in modelos.items():

    print("===================================")
    print("MODELO:", nombre)
    print("===================================")
    print()

    # Entrenar
    modelo.fit(X_train, y_train)

    print("Modelo entrenado correctamente.")
    print()

    # Predicciones sobre validación
    predicciones = modelo.predict(X_validation)

    # Métricas
    accuracy = accuracy_score(
        y_validation,
        predicciones
    )

    precision = precision_score(
        y_validation,
        predicciones,
        zero_division=0
    )

    recall = recall_score(
        y_validation,
        predicciones,
        zero_division=0
    )

    f1 = f1_score(
        y_validation,
        predicciones,
        zero_division=0
    )

    matriz = confusion_matrix(
        y_validation,
        predicciones
    )

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1-score :", round(f1, 4))
    print()

    print("Matriz de confusión:")
    print(matriz)
    print()

    resultados.append({
        "modelo": nombre,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })


# ============================================================
# COMPARACIÓN DE MODELOS
# ============================================================

resultados_df = pd.DataFrame(resultados)

resultados_df = resultados_df.sort_values(
    by="f1_score",
    ascending=False
)


print()
print("===================================")
print("COMPARACIÓN DE MODELOS")
print("===================================")
print()

print(
    resultados_df.to_string(
        index=False
    )
)

print()


# ============================================================
# GUARDAR RESULTADOS
# ============================================================

ARCHIVO_RESULTADOS = (
    f"{CARPETA_RESULTADOS}/comparacion_modelos.csv"
)

resultados_df.to_csv(
    ARCHIVO_RESULTADOS,
    index=False,
    encoding="utf-8"
)


# ============================================================
# MEJOR MODELO
# ============================================================

mejor_modelo = resultados_df.iloc[0]

print("===================================")
print("MEJOR MODELO")
print("===================================")
print()

print("Modelo:", mejor_modelo["modelo"])
print("Accuracy:", round(mejor_modelo["accuracy"], 4))
print("Precision:", round(mejor_modelo["precision"], 4))
print("Recall:", round(mejor_modelo["recall"], 4))
print("F1-score:", round(mejor_modelo["f1_score"], 4))
print()

print("Resultados guardados en:")
print(ARCHIVO_RESULTADOS)
print()

print("===================================")
print("PROCESO TERMINADO")
print("===================================")