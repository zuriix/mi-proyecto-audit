
import pandas as pd
from pathlib import Path

ARCHIVO = Path("data/processed/dataset_ml.csv")

CARACTERISTICAS = [
    "edad_version_dias",
    "versiones_anteriores",
    "numero_dependencias",
    "numero_dependencias_opcionales",
    "numero_scripts",
    "tiene_repositorio",
    "tiene_licencia",
]

COLUMNAS_REQUERIDAS = [
    "paquete",
    "version",
    *CARACTERISTICAS,
    "riesgo",
]

def main():
    if not ARCHIVO.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo: {ARCHIVO}"
        )

    df = pd.read_csv(ARCHIVO)

    print("=" * 55)
    print("VALIDACIÓN DEL DATASET PARA MACHINE LEARNING")
    print("=" * 55)

    print(f"\nDimensiones: {df.shape}")
    print(f"Columnas: {df.columns.tolist()}")

    faltantes = [
        c for c in COLUMNAS_REQUERIDAS
        if c not in df.columns
    ]

    if faltantes:
        print(f"\nERROR: faltan columnas: {faltantes}")
        return

    print("\nValores nulos por columna:")
    print(df[COLUMNAS_REQUERIDAS].isnull().sum())

    print("\nDuplicados paquete-versión:")
    print(df.duplicated(["paquete", "version"]).sum())

    print("\nDistribución de riesgo:")
    print(df["riesgo"].value_counts().sort_index())

    print("\nValores únicos de riesgo:")
    print(sorted(df["riesgo"].dropna().unique().tolist()))

    print("\nEstadísticas de las características:")
    print(df[CARACTERISTICAS].describe().round(2))

    print("\nColumnas relacionadas con vulnerabilidades:")
    relacionadas = [
        c for c in df.columns
        if "vulnerabilidad" in c.lower()
        or "vulnerabilities" in c.lower()
    ]
    print(relacionadas if relacionadas else "Ninguna")

    problemas = []

    if df.empty:
        problemas.append("El dataset está vacío.")

    if faltantes:
        problemas.append("Faltan columnas requeridas.")

    if df[["paquete", "version", "riesgo"]].isnull().any().any():
        problemas.append("Hay nulos en identificadores o etiquetas.")

    if df.duplicated(["paquete", "version"]).any():
        problemas.append("Hay versiones duplicadas.")

    if not df["riesgo"].isin([0, 1]).all():
        problemas.append("Hay etiquetas distintas de 0 y 1.")

    if df["riesgo"].nunique() != 2:
        problemas.append("No están presentes ambas clases.")

    if df[CARACTERISTICAS].isnull().any().any():
        problemas.append("Hay nulos en las características.")

    if relacionadas:
        problemas.append(
            "Revisar posibles variables de fuga de información."
        )

    print("\n" + "=" * 55)
    if problemas:
        print("RESULTADO: REVISAR")
        for problema in problemas:
            print(f"- {problema}")
    else:
        print("RESULTADO: VALIDACIONES BÁSICAS SUPERADAS")
    print("=" * 55)

if __name__ == "__main__":
    main()

