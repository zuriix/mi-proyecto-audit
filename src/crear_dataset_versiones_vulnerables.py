import pandas as pd
import requests
import time

ARCHIVO_ENTRADA = "data/raw/dataset_npm.csv"
ARCHIVO_SALIDA = "data/raw/dataset_versiones_vulnerables.csv"

URL_OSV = "https://api.osv.dev/v1/query"


def consultar_version(paquete, version):
    consulta = {
        "package": {
            "name": paquete,
            "ecosystem": "npm"
        },
        "version": version
    }

    try:
        respuesta = requests.post(
            URL_OSV,
            json=consulta,
            timeout=20
        )

        if respuesta.status_code == 200:
            datos = respuesta.json()
            return datos.get("vulns", [])

        print(
            f"Error OSV {paquete} {version}: "
            f"{respuesta.status_code}"
        )

    except requests.RequestException as error:
        print(
            f"Error de conexión {paquete} {version}: "
            f"{error}"
        )

    return None


def obtener_versiones_afectadas(paquete):
    consulta = {
        "package": {
            "name": paquete,
            "ecosystem": "npm"
        }
    }

    try:
        respuesta = requests.post(
            URL_OSV,
            json=consulta,
            timeout=20
        )

        if respuesta.status_code != 200:
            return []

        datos = respuesta.json()
        versiones = set()

        for vulnerabilidad in datos.get("vulns", []):

            for afectado in vulnerabilidad.get(
                "affected", []
            ):

                for rango in afectado.get(
                    "ranges", []
                ):

                    for evento in rango.get(
                        "events", []
                    ):

                        if "last_affected" in evento:
                            versiones.add(
                                evento["last_affected"]
                            )

        return sorted(versiones)

    except requests.RequestException:
        return []


df = pd.read_csv(ARCHIVO_ENTRADA)

resultados = []

for _, fila in df.iterrows():

    paquete = fila["paquete"]

    print("\n===================================")
    print("Paquete:", paquete)
    print("===================================")

    versiones = obtener_versiones_afectadas(
        paquete
    )

    print(
        "Versiones candidatas encontradas:",
        len(versiones)
    )

    for version in versiones:

        print(
            f"Comprobando {paquete} {version}"
        )

        vulnerabilidades = consultar_version(
            paquete,
            version
        )

        if vulnerabilidades is None:
            continue

        cantidad = len(vulnerabilidades)

        if cantidad > 0:

            resultados.append({
                "paquete": paquete,
                "version": version,
                "numero_vulnerabilidades": cantidad,
                "riesgo": 1
            })

            print(
                f"  OK -> riesgo=1 "
                f"({cantidad} vulnerabilidades)"
            )

        time.sleep(1)


dataset = pd.DataFrame(resultados)

if not dataset.empty:

    dataset = dataset.drop_duplicates(
        subset=["paquete", "version"]
    )

dataset.to_csv(
    ARCHIVO_SALIDA,
    index=False,
    encoding="utf-8"
)

print("\n===================================")
print("RECOLECCIÓN TERMINADA")
print("===================================")

print(
    "Registros vulnerables:",
    len(dataset)
)

print(
    "Archivo generado:",
    ARCHIVO_SALIDA
)

if not dataset.empty:

    print("\nDistribución:")
    print(
        dataset["riesgo"].value_counts()
    )