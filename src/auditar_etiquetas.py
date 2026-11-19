import pandas as pd

ARCHIVO = "data/processed/dataset_final.csv"

print()
print("===================================")
print("AUDITORÍA DE ETIQUETAS")
print("===================================")
print()

df = pd.read_csv(ARCHIVO)

paquetes = [
    "commander",
    "jsonwebtoken",
    "uuid"
]

for paquete in paquetes:

    print()
    print("===================================")
    print("PAQUETE:", paquete)
    print("===================================")
    print()

    datos = df[
        df["paquete"] == paquete
    ].copy()

    if datos.empty:
        print("No se encontraron registros.")
        continue

    datos = datos.sort_values(
        by="numero_vulnerabilidades",
        ascending=False
    )

    print(
        datos[
            [
                "paquete",
                "version",
                "numero_vulnerabilidades",
                "riesgo"
            ]
        ].to_string(index=False)
    )

    print()

    print("Distribución de riesgo:")

    print(
        datos["riesgo"]
        .value_counts()
        .sort_index()
    )

    print()

    print("Distribución de vulnerabilidades:")

    print(
        datos["numero_vulnerabilidades"]
        .value_counts()
        .sort_index()
    )

print()
print("===================================")
print("AUDITORÍA TERMINADA")
print("===================================")