import pandas as pd
import requests
import time

# Archivo de entrada
archivo_entrada = "data/raw/dataset_npm.csv"

# Archivo de salida
archivo_salida = "data/raw/dataset_npm_vulnerabilidades.csv"

# Leer dataset
df = pd.read_csv(archivo_entrada)

# Lista para guardar los resultados
vulnerabilidades_encontradas = []

# URL de la API de OSV
url = "https://api.osv.dev/v1/query"

for _, fila in df.iterrows():

    paquete = fila["paquete"]
    version = fila["version_actual"]

    print(f"Consultando OSV: {paquete} {version}")

    consulta = {
        "package": {
            "name": paquete,
            "ecosystem": "npm"
        },
        "version": version
    }

    try:

        respuesta = requests.post(
            url,
            json=consulta,
            timeout=15
        )

        if respuesta.status_code == 200:

            datos = respuesta.json()

            vulnerabilidades = datos.get("vulns", [])

            cantidad = len(vulnerabilidades)

            print(
                f"OK: {paquete} - "
                f"{cantidad} vulnerabilidades"
            )

        else:

            print(
                f"Error OSV en {paquete}: "
                f"{respuesta.status_code}"
            )

            cantidad = -1

    except requests.RequestException as error:

        print(
            f"Error de conexión con OSV "
            f"para {paquete}: {error}"
        )

        cantidad = -1

    vulnerabilidades_encontradas.append(cantidad)

    # Pausa para evitar demasiadas solicitudes seguidas
    time.sleep(1)


# Agregar nueva columna
df["numero_vulnerabilidades"] = vulnerabilidades_encontradas


# Crear archivo nuevo
df.to_csv(
    archivo_salida,
    index=False,
    encoding="utf-8"
)

print("\n===================================")
print("DATASET ACTUALIZADO")
print("===================================")

print("Paquetes procesados:", len(df))
print(
    "Archivo generado:",
    archivo_salida
)

print("\nResumen:")
print(
    df[
        [
            "paquete",
            "version_actual",
            "numero_vulnerabilidades"
        ]
    ].to_string(index=False)
)