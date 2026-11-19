import pandas as pd
import os

# ===================================
# CONFIGURACIÓN
# ===================================

ARCHIVO_PREDICCIONES = (
    "data/results/predicciones_test_temporal.csv"
)

CARPETA_RESULTADOS = "data/results"

os.makedirs(CARPETA_RESULTADOS, exist_ok=True)

print()
print("===================================")
print("ANÁLISIS DE PRUEBA TEMPORAL")
print("===================================")
print()

# ===================================
# CARGAR PREDICCIONES
# ===================================

df = pd.read_csv(ARCHIVO_PREDICCIONES)

print("Registros cargados:", len(df))
print()

# ===================================
# VERIFICAR COLUMNAS
# ===================================

columnas_necesarias = [
    "paquete",
    "version",
    "riesgo",
    "prediccion_riesgo"
]

columnas_faltantes = [
    columna
    for columna in columnas_necesarias
    if columna not in df.columns
]

if columnas_faltantes:
    print("ERROR: faltan columnas:")
    print(columnas_faltantes)
    raise SystemExit

# ===================================
# CREAR COLUMNA DE ACIERTO
# ===================================

df["acierto"] = (
    df["riesgo"] == df["prediccion_riesgo"]
)

# ===================================
# RESULTADOS GENERALES
# ===================================

print("===================================")
print("RESULTADO GENERAL")
print("===================================")
print()

total = len(df)
aciertos = df["acierto"].sum()
errores = total - aciertos

print("Total de registros:", total)
print("Predicciones correctas:", aciertos)
print("Predicciones incorrectas:", errores)

print(
    "Accuracy:",
    f"{aciertos / total * 100:.2f}%"
)

print()

# ===================================
# ANÁLISIS POR PAQUETE
# ===================================

print("===================================")
print("RESULTADOS POR PAQUETE")
print("===================================")
print()

resultados_paquete = (
    df.groupby("paquete")
    .agg(
        registros=("paquete", "size"),
        sin_riesgo_real=("riesgo", lambda x: (x == 0).sum()),
        con_riesgo_real=("riesgo", lambda x: (x == 1).sum()),
        aciertos=("acierto", "sum")
    )
    .reset_index()
)

resultados_paquete["errores"] = (
    resultados_paquete["registros"]
    - resultados_paquete["aciertos"]
)

resultados_paquete["accuracy"] = (
    resultados_paquete["aciertos"]
    / resultados_paquete["registros"]
)

resultados_paquete = resultados_paquete.sort_values(
    by="accuracy",
    ascending=True
)

print(
    resultados_paquete.to_string(
        index=False
    )
)

print()

# ===================================
# DISTRIBUCIÓN REAL
# ===================================

print("===================================")
print("DISTRIBUCIÓN REAL DE RIESGO")
print("===================================")
print()

print(
    df["riesgo"]
    .value_counts()
    .sort_index()
)

print()

# ===================================
# DISTRIBUCIÓN DE PREDICCIONES
# ===================================

print("===================================")
print("DISTRIBUCIÓN DE PREDICCIONES")
print("===================================")
print()

print(
    df["prediccion_riesgo"]
    .value_counts()
    .sort_index()
)

print()

# ===================================
# MATRIZ DE CONFUSIÓN MANUAL
# ===================================

verdaderos_negativos = (
    (df["riesgo"] == 0)
    & (df["prediccion_riesgo"] == 0)
).sum()

falsos_positivos = (
    (df["riesgo"] == 0)
    & (df["prediccion_riesgo"] == 1)
).sum()

falsos_negativos = (
    (df["riesgo"] == 1)
    & (df["prediccion_riesgo"] == 0)
).sum()

verdaderos_positivos = (
    (df["riesgo"] == 1)
    & (df["prediccion_riesgo"] == 1)
).sum()

print("===================================")
print("MATRIZ DE CONFUSIÓN")
print("===================================")
print()

print("Verdaderos negativos:", verdaderos_negativos)
print("Falsos positivos:", falsos_positivos)
print("Falsos negativos:", falsos_negativos)
print("Verdaderos positivos:", verdaderos_positivos)

print()

print("Formato:")
print()
print("                 Predicción")
print("                 0       1")
print("Real 0          ", verdaderos_negativos, "     ", falsos_positivos)
print("Real 1          ", falsos_negativos, "     ", verdaderos_positivos)

print()

# ===================================
# VERSIONES MAL CLASIFICADAS
# ===================================

print("===================================")
print("VERSIONES MAL CLASIFICADAS")
print("===================================")
print()

errores_df = df[
    df["acierto"] == False
].copy()

if len(errores_df) > 0:

    columnas_mostrar = [
        "paquete",
        "version",
        "riesgo",
        "prediccion_riesgo"
    ]

    print(
        errores_df[columnas_mostrar]
        .to_string(index=False)
    )

else:

    print("No se encontraron errores.")

print()

# ===================================
# GUARDAR RESULTADOS POR PAQUETE
# ===================================

ARCHIVO_PAQUETES = (
    f"{CARPETA_RESULTADOS}/"
    "analisis_test_temporal_paquetes.csv"
)

resultados_paquete.to_csv(
    ARCHIVO_PAQUETES,
    index=False,
    encoding="utf-8"
)

# ===================================
# GUARDAR ERRORES
# ===================================

ARCHIVO_ERRORES = (
    f"{CARPETA_RESULTADOS}/"
    "errores_test_temporal.csv"
)

errores_df.to_csv(
    ARCHIVO_ERRORES,
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

print(ARCHIVO_PAQUETES)
print(ARCHIVO_ERRORES)

print()

print("===================================")
print("ANÁLISIS TERMINADO")
print("===================================")