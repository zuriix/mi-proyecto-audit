from pathlib import Path
import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

BASE_DIR = Path(__file__).resolve().parent.parent
RUTA_DATOS = BASE_DIR / "data" / "processed"
RUTA_MODELOS = BASE_DIR / "models"

caracteristicas = [
    "edad_version_dias",
    "versiones_anteriores",
    "numero_dependencias",
    "numero_dependencias_opcionales",
    "numero_scripts",
    "tiene_repositorio",
    "tiene_licencia",
]

# Datos autorizados por el protocolo de evaluación final: train + validation
train = pd.read_csv(RUTA_DATOS / "train.csv")
validation = pd.read_csv(RUTA_DATOS / "validation.csv")
train_final = pd.concat([train, validation], ignore_index=True)

X_train_final = train_final[caracteristicas]
y_train_final = train_final["riesgo"]

# Configuración seleccionada mediante la comparación de validación en la Etapa 3
modelo = LogisticRegression(
    max_iter=2000,
    random_state=42,
)

print("Ajustando modelo seleccionado (LogisticRegression) con train + validation...")
modelo.fit(X_train_final, y_train_final)

os.makedirs(RUTA_MODELOS, exist_ok=True)

artefacto = {
    "modelo": modelo,
    "caracteristicas": caracteristicas,
    "algoritmo": "LogisticRegression",
    "hiperparametros": {"max_iter": 2000, "random_state": 42},
    "registros_entrenamiento": len(train_final),
    "paquetes_entrenamiento": train_final["paquete"].nunique(),
}

# Se guarda en models/random_forest.pkl para mantener compatibilidad estricta
# con app.py y src/detectar_riesgo.py sin modificar rutas de la aplicación existente.
ruta_artefacto_app = RUTA_MODELOS / "random_forest.pkl"
ruta_artefacto_etapa3 = RUTA_MODELOS / "modelo_final.pkl"

joblib.dump(artefacto, ruta_artefacto_app)
joblib.dump(artefacto, ruta_artefacto_etapa3)

print("Modelo guardado correctamente.")
print(f"Artefacto principal para la aplicación: {ruta_artefacto_app}")
print(f"Artefacto Etapa 3                     : {ruta_artefacto_etapa3}")
print(f"Algoritmo                             : LogisticRegression")
print(f"Registros de entrenamiento final      : {len(train_final)}")