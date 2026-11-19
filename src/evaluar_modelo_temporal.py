import pandas as pd
import os

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ===================================
# CONFIGURACIÓN
# ===================================

ARCHIVO_TRAIN = "data/processed/train_temporal.csv"
ARCHIVO_VALIDACION = "data/processed/validation_temporal.csv"
ARCHIVO_TEST = "data/processed/test_temporal.csv"

CARPETA_RESULTADOS = "data/results"

RANDOM_STATE = 42

os.makedirs(CARPETA_RESULTADOS, exist_ok=True)

print()
print("===================================")
print("EVALUACIÓN FINAL TEMPORAL")
print("===================================")
print()

# ===================================
# CARGAR DATOS
# ===================================

train = pd.read_csv(ARCHIVO_TRAIN)
validation = pd.read_csv(ARCHIVO_VALIDACION)
test = pd.read_csv(ARCHIVO_TEST)

print("Entrenamiento:", len(train))
print("Validación:", len(validation))
print("Prueba temporal:", len(test))
print()

# ===================================
# COMBINAR ENTRENAMIENTO + VALIDACIÓN
# ===================================
#
# La validación ya fue utilizada para
# seleccionar el mejor modelo.
#
# Por eso ahora podemos utilizar
# entrenamiento + validación para ajustar
# el modelo definitivo.
#
# El test temporal permanece separado.
# ===================================

train_final = pd.concat(
    [train, validation],
    ignore_index=True
)

print(
    "Datos utilizados para entrenar el modelo final:",
    len(train_final)
)

print(
    "Datos reservados para prueba final:",
    len(test)
)

print()

# ===================================
# CARACTERÍSTICAS
# ===================================

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

# ===================================
# VERIFICAR COLUMNAS
# ===================================

columnas_faltantes = [
    columna
    for columna in CARACTERISTICAS + [VARIABLE_OBJETIVO]
    if columna not in train_final.columns
    or columna not in test.columns
]

if columnas_faltantes:
    print("ERROR: faltan columnas:")
    print(columnas_faltantes)
    raise SystemExit

# ===================================
# PREPARAR X E Y
# ===================================

X_train_final = train_final[CARACTERISTICAS]
y_train_final = train_final[VARIABLE_OBJETIVO]

X_test = test[CARACTERISTICAS]
y_test = test[VARIABLE_OBJETIVO]

print("Características utilizadas:")

for caracteristica in CARACTERISTICAS:
    print("-", caracteristica)

print()

print("Variable objetivo:", VARIABLE_OBJETIVO)
print()

# ===================================
# CREAR MODELO FINAL
# ===================================

modelo = DecisionTreeClassifier(
    max_depth=5,
    random_state=RANDOM_STATE
)

print("===================================")
print("ENTRENANDO ÁRBOL DE DECISIÓN")
print("===================================")
print()

modelo.fit(
    X_train_final,
    y_train_final
)

print("Modelo entrenado correctamente.")
print()

# ===================================
# PREDICCIÓN SOBRE TEST TEMPORAL
# ===================================

predicciones = modelo.predict(X_test)

# ===================================
# MÉTRICAS
# ===================================

accuracy = accuracy_score(
    y_test,
    predicciones
)

precision = precision_score(
    y_test,
    predicciones,
    zero_division=0
)

recall = recall_score(
    y_test,
    predicciones,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predicciones,
    zero_division=0
)

matriz = confusion_matrix(
    y_test,
    predicciones
)

# ===================================
# MOSTRAR RESULTADOS
# ===================================

print("===================================")
print("RESULTADOS DE LA PRUEBA TEMPORAL")
print("===================================")
print()

print(
    "Accuracy :",
    round(accuracy, 4),
    f"({accuracy * 100:.2f}%)"
)

print(
    "Precision:",
    round(precision, 4),
    f"({precision * 100:.2f}%)"
)

print(
    "Recall   :",
    round(recall, 4),
    f"({recall * 100:.2f}%)"
)

print(
    "F1-score :",
    round(f1, 4),
    f"({f1 * 100:.2f}%)"
)

print()

print("Matriz de confusión:")
print(matriz)
print()

# ===================================
# REPORTE DE CLASIFICACIÓN
# ===================================

print("===================================")
print("REPORTE DE CLASIFICACIÓN")
print("===================================")
print()

print(
    classification_report(
        y_test,
        predicciones,
        target_names=[
            "Sin riesgo",
            "Con riesgo"
        ],
        zero_division=0
    )
)

# ===================================
# GUARDAR MÉTRICAS
# ===================================

resultados = pd.DataFrame([
    {
        "modelo": "Arbol de Decision",
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
])

ARCHIVO_RESULTADOS = (
    f"{CARPETA_RESULTADOS}/"
    "evaluacion_final_temporal.csv"
)

resultados.to_csv(
    ARCHIVO_RESULTADOS,
    index=False,
    encoding="utf-8"
)

# ===================================
# GUARDAR PREDICCIONES
# ===================================

predicciones_test = test.copy()

predicciones_test[
    "prediccion_riesgo"
] = predicciones

ARCHIVO_PREDICCIONES = (
    f"{CARPETA_RESULTADOS}/"
    "predicciones_test_temporal.csv"
)

predicciones_test.to_csv(
    ARCHIVO_PREDICCIONES,
    index=False,
    encoding="utf-8"
)

# ===================================
# FINAL
# ===================================

print("===================================")
print("ARCHIVOS GENERADOS")
print("===================================")
print()

print(ARCHIVO_RESULTADOS)
print(ARCHIVO_PREDICCIONES)
print()

print("===================================")
print("EVALUACIÓN FINALIZADA")
print("===================================")