import pandas as pd
import os

from sklearn.model_selection import train_test_split


# ==========================================
# ARCHIVOS
# ==========================================

ARCHIVO_ENTRADA = "data/processed/dataset_ml.csv"

ARCHIVO_TRAIN = "data/processed/train.csv"
ARCHIVO_VALIDACION = "data/processed/validation.csv"
ARCHIVO_TEST = "data/processed/test.csv"


# ==========================================
# CONFIGURACIÓN
# ==========================================

RANDOM_STATE = 42


os.makedirs(
    "data/processed",
    exist_ok=True
)


# ==========================================
# INICIO
# ==========================================

print()
print("===================================")
print("PREPARANDO DATOS PARA MACHINE LEARNING")
print("===================================")
print()


# ==========================================
# CARGAR DATASET
# ==========================================

dataset = pd.read_csv(
    ARCHIVO_ENTRADA
)


print(
    "Registros totales:",
    len(dataset)
)

print()


# ==========================================
# VERIFICAR COLUMNAS
# ==========================================

columnas_esperadas = [
    "paquete",
    "version",
    "edad_version_dias",
    "versiones_anteriores",
    "numero_dependencias",
    "numero_dependencias_opcionales",
    "numero_scripts",
    "tiene_repositorio",
    "tiene_licencia",
    "riesgo"
]


columnas_faltantes = [
    columna
    for columna in columnas_esperadas
    if columna not in dataset.columns
]


if columnas_faltantes:

    print(
        "ERROR: faltan las siguientes columnas:"
    )

    print(
        columnas_faltantes
    )

    raise SystemExit


# ==========================================
# ELIMINAR DUPLICADOS
# ==========================================

dataset = dataset.drop_duplicates(
    subset=[
        "paquete",
        "version"
    ]
)


# ==========================================
# ELIMINAR DATOS INCOMPLETOS
# ==========================================

dataset = dataset.dropna(
    subset=[
        "edad_version_dias",
        "versiones_anteriores",
        "numero_dependencias",
        "numero_dependencias_opcionales",
        "numero_scripts",
        "tiene_repositorio",
        "tiene_licencia",
        "riesgo"
    ]
)


# ==========================================
# ASEGURAR CLASES 0 Y 1
# ==========================================

dataset = dataset[
    dataset["riesgo"].isin([0, 1])
].copy()


# ==========================================
# MOSTRAR DISTRIBUCIÓN
# ==========================================

print("Distribución de clases:")

print(
    dataset["riesgo"].value_counts()
    .sort_index()
)

print()


# ==========================================
# PRIMERA SEPARACIÓN
#
# 80% entrenamiento + validación
# 20% prueba final
# ==========================================

train_validacion, test = train_test_split(
    dataset,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=dataset["riesgo"]
)


# ==========================================
# SEGUNDA SEPARACIÓN
#
# Del 80%:
# 75% entrenamiento
# 25% validación
#
# Resultado final:
# 60% entrenamiento
# 20% validación
# 20% prueba
# ==========================================

train, validation = train_test_split(
    train_validacion,
    test_size=0.25,
    random_state=RANDOM_STATE,
    stratify=train_validacion["riesgo"]
)


# ==========================================
# GUARDAR DATASETS
# ==========================================

train.to_csv(
    ARCHIVO_TRAIN,
    index=False,
    encoding="utf-8"
)


validation.to_csv(
    ARCHIVO_VALIDACION,
    index=False,
    encoding="utf-8"
)


test.to_csv(
    ARCHIVO_TEST,
    index=False,
    encoding="utf-8"
)


# ==========================================
# MOSTRAR RESULTADOS
# ==========================================

print("===================================")
print("SEPARACIÓN DE DATOS COMPLETADA")
print("===================================")
print()


print(
    "Entrenamiento:",
    len(train),
    "registros"
)

print(
    "Validación:",
    len(validation),
    "registros"
)

print(
    "Prueba final:",
    len(test),
    "registros"
)

print()


# ==========================================
# DISTRIBUCIÓN DE CADA CONJUNTO
# ==========================================

print("Distribución de entrenamiento:")

print(
    train["riesgo"]
    .value_counts()
    .sort_index()
)

print()


print("Distribución de validación:")

print(
    validation["riesgo"]
    .value_counts()
    .sort_index()
)

print()


print("Distribución de prueba:")

print(
    test["riesgo"]
    .value_counts()
    .sort_index()
)

print()


# ==========================================
# PORCENTAJES
# ==========================================

total = len(dataset)

print("Porcentajes:")

print(
    f"Entrenamiento: "
    f"{len(train) / total * 100:.2f}%"
)

print(
    f"Validación: "
    f"{len(validation) / total * 100:.2f}%"
)

print(
    f"Prueba: "
    f"{len(test) / total * 100:.2f}%"
)

print()


# ==========================================
# ARCHIVOS GENERADOS
# ==========================================

print("Archivos generados:")

print(
    ARCHIVO_TRAIN
)

print(
    ARCHIVO_VALIDACION
)

print(
    ARCHIVO_TEST
)

print()


print("===================================")
print("PROCESO TERMINADO")
print("===================================")