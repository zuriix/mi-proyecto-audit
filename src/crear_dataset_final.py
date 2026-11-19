import pandas as pd
import os


# ==========================================
# ARCHIVOS
# ==========================================

ARCHIVO_VULNERABLES = (
    "data/raw/dataset_versiones_vulnerables_v2.csv"
)

ARCHIVO_NO_VULNERABLES = (
    "data/raw/dataset_versiones_no_vulnerables.csv"
)

ARCHIVO_SALIDA = (
    "data/processed/dataset_final.csv"
)


# ==========================================
# CREAR CARPETA DE SALIDA
# ==========================================

os.makedirs(
    "data/processed",
    exist_ok=True
)


# ==========================================
# INICIO
# ==========================================

print()
print("===================================")
print("CREANDO DATASET FINAL")
print("===================================")
print()


# ==========================================
# LEER DATASETS
# ==========================================

try:

    df_vulnerables = pd.read_csv(
        ARCHIVO_VULNERABLES
    )

except FileNotFoundError:

    print(
        "ERROR: No se encontró el archivo:"
    )

    print(
        ARCHIVO_VULNERABLES
    )

    raise


try:

    df_no_vulnerables = pd.read_csv(
        ARCHIVO_NO_VULNERABLES
    )

except FileNotFoundError:

    print(
        "ERROR: No se encontró el archivo:"
    )

    print(
        ARCHIVO_NO_VULNERABLES
    )

    raise


# ==========================================
# MOSTRAR CANTIDAD DE REGISTROS
# ==========================================

print(
    "Registros vulnerables:",
    len(df_vulnerables)
)

print(
    "Registros no vulnerables:",
    len(df_no_vulnerables)
)


# ==========================================
# PREPARAR DATASET VULNERABLE
# ==========================================

df_vulnerables = df_vulnerables[
    [
        "paquete",
        "version",
        "numero_vulnerabilidades",
        "riesgo"
    ]
].copy()


# ==========================================
# PREPARAR DATASET NO VULNERABLE
# ==========================================

df_no_vulnerables = df_no_vulnerables[
    [
        "paquete",
        "version",
        "numero_vulnerabilidades",
        "riesgo"
    ]
].copy()


# ==========================================
# ASEGURAR TIPOS DE DATOS
# ==========================================

df_vulnerables["riesgo"] = (
    pd.to_numeric(
        df_vulnerables["riesgo"],
        errors="coerce"
    )
)

df_no_vulnerables["riesgo"] = (
    pd.to_numeric(
        df_no_vulnerables["riesgo"],
        errors="coerce"
    )
)


df_vulnerables["numero_vulnerabilidades"] = (
    pd.to_numeric(
        df_vulnerables[
            "numero_vulnerabilidades"
        ],
        errors="coerce"
    )
)

df_no_vulnerables["numero_vulnerabilidades"] = (
    pd.to_numeric(
        df_no_vulnerables[
            "numero_vulnerabilidades"
        ],
        errors="coerce"
    )
)


# ==========================================
# UNIR DATASETS
# ==========================================

dataset = pd.concat(
    [
        df_vulnerables,
        df_no_vulnerables
    ],
    ignore_index=True
)


# ==========================================
# ELIMINAR REGISTROS DUPLICADOS
# ==========================================

antes = len(dataset)

dataset = dataset.drop_duplicates(
    subset=[
        "paquete",
        "version"
    ]
)

despues = len(dataset)

duplicados = antes - despues


print()
print(
    "Registros duplicados eliminados:",
    duplicados
)


# ==========================================
# ELIMINAR REGISTROS CON DATOS FALTANTES
# ==========================================

antes = len(dataset)

dataset = dataset.dropna(
    subset=[
        "paquete",
        "version",
        "numero_vulnerabilidades",
        "riesgo"
    ]
)

despues = len(dataset)

faltantes_eliminados = antes - despues


print(
    "Registros con datos faltantes eliminados:",
    faltantes_eliminados
)


# ==========================================
# ASEGURAR QUE RIESGO SEA 0 O 1
# ==========================================

dataset = dataset[
    dataset["riesgo"].isin(
        [0, 1]
    )
].copy()


# ==========================================
# ORDENAR DATASET
# ==========================================

dataset = dataset.sort_values(
    by=[
        "paquete",
        "version"
    ]
)


# ==========================================
# REINICIAR ÍNDICES
# ==========================================

dataset = dataset.reset_index(
    drop=True
)


# ==========================================
# VALORES FALTANTES
# ==========================================

print()
print("===================================")
print("VALORES FALTANTES")
print("===================================")

print(
    dataset.isnull().sum()
)


# ==========================================
# DISTRIBUCIÓN DE RIESGO
# ==========================================

print()
print("===================================")
print("DISTRIBUCIÓN DE RIESGO")
print("===================================")

distribucion = (
    dataset["riesgo"]
    .value_counts()
    .sort_index()
)

print(
    distribucion
)


# ==========================================
# PORCENTAJES DE RIESGO
# ==========================================

print()
print("===================================")
print("PORCENTAJES DE RIESGO")
print("===================================")

porcentajes = (
    dataset["riesgo"]
    .value_counts(
        normalize=True
    )
    .sort_index()
    * 100
)

for riesgo, porcentaje in porcentajes.items():

    print(
        f"Riesgo {int(riesgo)}: "
        f"{porcentaje:.2f}%"
    )


# ==========================================
# REGISTROS POR PAQUETE
# ==========================================

print()
print("===================================")
print("REGISTROS POR PAQUETE")
print("===================================")

print(
    dataset["paquete"]
    .value_counts()
)


# ==========================================
# DISTRIBUCIÓN POR PAQUETE Y RIESGO
# ==========================================

print()
print("===================================")
print("RIESGO POR PAQUETE")
print("===================================")

tabla_riesgo = pd.crosstab(
    dataset["paquete"],
    dataset["riesgo"]
)

print(
    tabla_riesgo
)


# ==========================================
# ESTADÍSTICAS DE VULNERABILIDADES
# ==========================================

print()
print("===================================")
print("ESTADÍSTICAS DE VULNERABILIDADES")
print("===================================")

print(
    dataset[
        "numero_vulnerabilidades"
    ].describe()
)


# ==========================================
# INFORMACIÓN DEL DATASET
# ==========================================

print()
print("===================================")
print("INFORMACIÓN DEL DATASET FINAL")
print("===================================")

print(
    "Número de registros:",
    len(dataset)
)

print(
    "Número de columnas:",
    len(dataset.columns)
)

print()
print("Columnas:")

print(
    list(dataset.columns)
)


# ==========================================
# GUARDAR DATASET
# ==========================================

dataset.to_csv(
    ARCHIVO_SALIDA,
    index=False,
    encoding="utf-8"
)


# ==========================================
# RESULTADO FINAL
# ==========================================

print()
print("===================================")
print("DATASET FINAL GENERADO")
print("===================================")

print(
    "Archivo:",
    ARCHIVO_SALIDA
)

print(
    "Total de registros:",
    len(dataset)
)

print()
print("===================================")
print("PROCESO TERMINADO")
print("===================================")