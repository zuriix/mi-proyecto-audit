import pandas as pd
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

ARCHIVO_TRAIN = "data/processed/train.csv"
ARCHIVO_VALIDACION = "data/processed/validation.csv"
ARCHIVO_TEST = "data/processed/test.csv"

CARPETA_RESULTADOS = "data/results"

RANDOM_STATE = 42

os.makedirs(CARPETA_RESULTADOS, exist_ok=True)

print()
print("===================================")
print("EVALUACIÓN FINAL DEL MODELO")
print("===================================")
print()

# -----------------------------------
# CARGAR DATOS
# -----------------------------------

train = pd.read_csv(ARCHIVO_TRAIN)
validation = pd.read_csv(ARCHIVO_VALIDACION)
test = pd.read_csv(ARCHIVO_TEST)

print("Registros de entrenamiento:", len(train))
print("Registros de validación:", len(validation))
print("Registros de prueba:", len(test))
print()

# -----------------------------------
# CARACTERÍSTICAS
# -----------------------------------

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

# -----------------------------------
# COMBINAR TRAIN + VALIDACIÓN
# -----------------------------------

train_final = pd.concat(
    [train, validation],
    ignore_index=True
)

print("Datos utilizados para entrenar el modelo final:",
      len(train_final))
print()

X_train_final = train_final[CARACTERISTICAS]
y_train_final = train_final[VARIABLE_OBJETIVO]

X_test = test[CARACTERISTICAS]
y_test = test[VARIABLE_OBJETIVO]

# -----------------------------------
# CREAR MODELO
# -----------------------------------

modelo = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=RANDOM_STATE,
    class_weight="balanced"
)

print("Entrenando Random Forest...")
print()

modelo.fit(X_train_final, y_train_final)

print("Modelo entrenado correctamente.")
print()

# -----------------------------------
# EVALUACIÓN FINAL
# -----------------------------------

predicciones = modelo.predict(X_test)

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

# -----------------------------------
# MOSTRAR RESULTADOS
# -----------------------------------

print("===================================")
print("RESULTADOS EN EL CONJUNTO DE PRUEBA")
print("===================================")
print()

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print()

print("Porcentajes:")
print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall   : {recall * 100:.2f}%")
print(f"F1-score : {f1 * 100:.2f}%")
print()

print("Matriz de confusión:")
print(matriz)
print()

# -----------------------------------
# GUARDAR RESULTADOS
# -----------------------------------

resultados = pd.DataFrame([
    {
        "modelo": "Random Forest",
        "conjunto": "test",
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
])

ARCHIVO_RESULTADOS = (
    f"{CARPETA_RESULTADOS}/evaluacion_final.csv"
)

resultados.to_csv(
    ARCHIVO_RESULTADOS,
    index=False,
    encoding="utf-8"
)

print("Resultados guardados en:")
print(ARCHIVO_RESULTADOS)
print()

print("===================================")
print("EVALUACIÓN FINAL COMPLETADA")
print("===================================")