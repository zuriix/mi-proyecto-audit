import pandas as pd

ARCHIVO = "data/processed/dataset_final.csv"


print("===================================")
print(" ANÁLISIS DEL DATASET FINAL")
print("===================================")


# ==========================================
# 1. CARGAR DATASET
# ==========================================

df = pd.read_csv(ARCHIVO)


# ==========================================
# 2. INFORMACIÓN GENERAL
# ==========================================

print("\n===================================")
print("INFORMACIÓN GENERAL")
print("===================================")

print("Número de registros:", len(df))
print("Número de columnas:", len(df.columns))

print("\nColumnas:")
print(list(df.columns))


# ==========================================
# 3. VALORES FALTANTES
# ==========================================

print("\n===================================")
print("VALORES FALTANTES")
print("===================================")

print(df.isnull().sum())


# ==========================================
# 4. DISTRIBUCIÓN DE RIESGO
# ==========================================

print("\n===================================")
print("DISTRIBUCIÓN DE RIESGO")
print("===================================")

distribucion = df["riesgo"].value_counts()

print(distribucion)


# ==========================================
# 5. PORCENTAJES DE RIESGO
# ==========================================

print("\n===================================")
print("PORCENTAJES DE RIESGO")
print("===================================")

porcentajes = (
    df["riesgo"]
    .value_counts(normalize=True)
    * 100
)

print(porcentajes.round(2))


# ==========================================
# 6. REGISTROS POR PAQUETE
# ==========================================

print("\n===================================")
print("REGISTROS POR PAQUETE")
print("===================================")

print(
    df["paquete"].value_counts()
)


# ==========================================
# 7. RIESGO POR PAQUETE
# ==========================================

print("\n===================================")
print("RIESGO POR PAQUETE")
print("===================================")

tabla_riesgo = pd.crosstab(
    df["paquete"],
    df["riesgo"]
)

print(tabla_riesgo)


# ==========================================
# 8. ESTADÍSTICAS DE VULNERABILIDADES
# ==========================================

print("\n===================================")
print("ESTADÍSTICAS DE VULNERABILIDADES")
print("===================================")

print(
    df["numero_vulnerabilidades"].describe()
)


# ==========================================
# 9. VULNERABILIDADES POR PAQUETE
# ==========================================

print("\n===================================")
print("VULNERABILIDADES POR PAQUETE")
print("===================================")

print(
    df.groupby("paquete")[
        "numero_vulnerabilidades"
    ].agg(
        [
            "count",
            "min",
            "max",
            "mean"
        ]
    )
)


# ==========================================
# 10. CONCLUSIÓN
# ==========================================

print("\n===================================")
print("CONCLUSIÓN DEL ANÁLISIS")
print("===================================")

cantidad_riesgo_0 = (
    (df["riesgo"] == 0).sum()
)

cantidad_riesgo_1 = (
    (df["riesgo"] == 1).sum()
)

print(
    f"Registros sin riesgo: {cantidad_riesgo_0}"
)

print(
    f"Registros con riesgo: {cantidad_riesgo_1}"
)

if cantidad_riesgo_0 < cantidad_riesgo_1:

    print(
        "\nSe identifica un desbalance "
        "entre las clases."
    )

    print(
        "Antes del entrenamiento será "
        "necesario preparar los datos."
    )

else:

    print(
        "\nLas clases no presentan "
        "un desbalance dominante."
    )


print("\n===================================")
print("ANÁLISIS TERMINADO")
print("===================================")