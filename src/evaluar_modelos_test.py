
import os
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

RUTA = "data/processed"

train = pd.read_csv(f"{RUTA}/train.csv")
test = pd.read_csv(f"{RUTA}/test.csv")

caracteristicas = [
    "edad_version_dias",
    "versiones_anteriores",
    "numero_dependencias",
    "numero_dependencias_opcionales",
    "numero_scripts",
    "tiene_repositorio",
    "tiene_licencia",
]

X_train = train[caracteristicas]
y_train = train["riesgo"]

X_test = test[caracteristicas]
y_test = test["riesgo"]

modelos = {
    "Regresion Logistica": LogisticRegression(max_iter=2000),
    "Arbol de Decision": DecisionTreeClassifier(
        max_depth=5,
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        class_weight="balanced",
    ),
}

resultados = []

print("=" * 55)
print("EVALUACION FINAL EN DATOS DE PRUEBA")
print("=" * 55)
print(f"Entrenamiento: {len(train)} registros")
print(f"Prueba: {len(test)} registros")

for nombre, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    predicciones = modelo.predict(X_test)

    accuracy = accuracy_score(y_test, predicciones)
    precision = precision_score(
        y_test, predicciones, zero_division=0
    )
    recall = recall_score(
        y_test, predicciones, zero_division=0
    )
    f1 = f1_score(
        y_test, predicciones, zero_division=0
    )

    matriz = confusion_matrix(
        y_test, predicciones, labels=[0, 1]
    )

    print(f"\nMODELO: {nombre}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("Matriz de confusión [ [TN, FP], [FN, TP] ]:")
    print(matriz)
    print(classification_report(
        y_test,
        predicciones,
        labels=[0, 1],
        target_names=["Riesgo 0", "Riesgo 1"],
        zero_division=0,
    ))

    resultados.append({
        "modelo": nombre,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "tn": int(matriz[0, 0]),
        "fp": int(matriz[0, 1]),
        "fn": int(matriz[1, 0]),
        "tp": int(matriz[1, 1]),
    })

os.makedirs("data/results", exist_ok=True)

salida = "data/results/evaluacion_final_test.csv"
pd.DataFrame(resultados).to_csv(salida, index=False)

print("=" * 55)
print(f"Resultados guardados en: {salida}")
print("=" * 55)
