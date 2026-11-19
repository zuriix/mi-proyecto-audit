import pandas as pd
import os

ARCHIVO_ENTRADA = "data/processed/dataset_ml.csv"

ARCHIVO_TRAIN = "data/processed/train_temporal.csv"
ARCHIVO_VALIDACION = "data/processed/validation_temporal.csv"
ARCHIVO_TEST = "data/processed/test_temporal.csv"

os.makedirs("data/processed", exist_ok=True)

print()
print("===================================")
print("PREPARACIÓN DE DATOS TEMPORAL")
print("===================================")
print()

# -----------------------------------
# CARGAR DATASET
# -----------------------------------

dataset = pd.read_csv(ARCHIVO_ENTRADA)

print("Registros originales:", len(dataset))
print()

# -----------------------------------
# VALIDAR COLUMNAS
# -----------------------------------

columnas_necesarias = [
    "paquete",
    "version",
    "edad_version_dias",
    "riesgo"
]

columnas_faltantes = [
    columna
    for columna in columnas_necesarias
    if columna not in dataset.columns
]

if columnas_faltantes:
    print("ERROR: faltan columnas:")
    print(columnas_faltantes)
    raise SystemExit

# -----------------------------------
# LIMPIAR DATOS
# -----------------------------------

dataset = dataset.drop_duplicates(
    subset=["paquete", "version"]
)

dataset = dataset.dropna(
    subset=[
        "paquete",
        "version",
        "edad_version_dias",
        "riesgo"
    ]
)

dataset = dataset[
    dataset["riesgo"].isin([0, 1])
].copy()

# -----------------------------------
# ORDEN TEMPORAL
# -----------------------------------
#
# edad_version_dias:
#
# mayor valor = versión más antigua
# menor valor = versión más reciente
#
# Por eso ordenamos de mayor a menor.
# -----------------------------------

dataset = dataset.sort_values(
    by=["paquete", "edad_version_dias"],
    ascending=[True, False]
).reset_index(drop=True)

# -----------------------------------
# SEPARACIÓN POR PAQUETE
# -----------------------------------
#
# Para cada paquete:
#
# 60 % versiones antiguas -> TRAIN
# 20 % siguientes           -> VALIDATION
# 20 % más recientes        -> TEST
#
# -----------------------------------

train_partes = []
validation_partes = []
test_partes = []

for paquete, grupo in dataset.groupby(
    "paquete",
    sort=False
):

    grupo = grupo.sort_values(
        by="edad_version_dias",
        ascending=False
    ).reset_index(drop=True)

    total = len(grupo)

    limite_train = int(total * 0.60)
    limite_validation = int(total * 0.80)

    # Garantizar al menos un registro
    if limite_train < 1:
        limite_train = 1

    if limite_validation <= limite_train:
        limite_validation = limite_train + 1

    train_grupo = grupo.iloc[:limite_train]

    validation_grupo = grupo.iloc[
        limite_train:limite_validation
    ]

    test_grupo = grupo.iloc[
        limite_validation:
    ]

    train_partes.append(train_grupo)
    validation_partes.append(validation_grupo)
    test_partes.append(test_grupo)

# -----------------------------------
# UNIR GRUPOS
# -----------------------------------

train = pd.concat(
    train_partes,
    ignore_index=True
)

validation = pd.concat(
    validation_partes,
    ignore_index=True
)

test = pd.concat(
    test_partes,
    ignore_index=True
)

# -----------------------------------
# ORDENAR LOS ARCHIVOS
# -----------------------------------

train = train.sort_values(
    by=["paquete", "edad_version_dias"],
    ascending=[True, False]
)

validation = validation.sort_values(
    by=["paquete", "edad_version_dias"],
    ascending=[True, False]
)

test = test.sort_values(
    by=["paquete", "edad_version_dias"],
    ascending=[True, False]
)

# -----------------------------------
# GUARDAR
# -----------------------------------

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

# -----------------------------------
# MOSTRAR RESULTADOS
# -----------------------------------

print("===================================")
print("SEPARACIÓN TEMPORAL COMPLETADA")
print("===================================")
print()

print("Total:", len(dataset))
print("Entrenamiento:", len(train))
print("Validación:", len(validation))
print("Prueba temporal:", len(test))
print()

print("Porcentajes:")
print(
    f"Entrenamiento: "
    f"{len(train) / len(dataset) * 100:.2f}%"
)

print(
    f"Validación: "
    f"{len(validation) / len(dataset) * 100:.2f}%"
)

print(
    f"Prueba: "
    f"{len(test) / len(dataset) * 100:.2f}%"
)

print()

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

print("Archivos generados:")
print(ARCHIVO_TRAIN)
print(ARCHIVO_VALIDACION)
print(ARCHIVO_TEST)
print()

print("===================================")
print("PROCESO TERMINADO")
print("===================================")