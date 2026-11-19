import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

RUTA_DATOS = "data/processed"
RUTA_MODELOS = "models"

caracteristicas = [
    "edad_version_dias",
    "versiones_anteriores",
    "numero_dependencias",
    "numero_dependencias_opcionales",
    "numero_scripts",
    "tiene_repositorio",
    "tiene_licencia",
]

train = pd.read_csv(f"{RUTA_DATOS}/train.csv")

X_train = train[caracteristicas]
y_train = train["riesgo"]

modelo = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42,
    class_weight="balanced",
)

modelo.fit(X_train, y_train)

os.makedirs(RUTA_MODELOS, exist_ok=True)

joblib.dump(
    {"modelo": modelo, "caracteristicas": caracteristicas},
    f"{RUTA_MODELOS}/random_forest.pkl",
)

print("Modelo guardado correctamente.")
print(f"Archivo: {RUTA_MODELOS}/random_forest.pkl")
print(f"Registros de entrenamiento: {len(train)}")