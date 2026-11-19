
"""
OSS Guard - Construcción del dataset etiquetado con npm Registry y OSV.

Ejecutar desde la carpeta raíz:
    python src/crear_dataset_etiquetado_osv.py

Reglas:
    riesgo = 0: OSV devuelve cero vulnerabilidades conocidas.
    riesgo = 1: OSV devuelve una o más vulnerabilidades conocidas.

Si la consulta falla, la versión queda registrada como error;
nunca se etiqueta automáticamente como segura.
"""

import os
import re
import time
import random
from urllib.parse import quote

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ============================================================
# CONFIGURACIÓN
# ============================================================

ARCHIVO_ENTRADA = "data/raw/dataset_npm.csv"
ARCHIVO_SALIDA = "data/raw/dataset_etiquetado_osv.csv"
ARCHIVO_DIAGNOSTICO = "data/raw/diagnostico_etiquetado_osv.csv"

LIMITE_VERSIONES_POR_PAQUETE = 100

URL_NPM = "https://registry.npmjs.org"
URL_OSV = "https://api.osv.dev/v1/query"

TIMEOUT = 30
PAUSA_OSV = 0.10

os.makedirs("data/raw", exist_ok=True)

sesion = requests.Session()
sesion.headers.update({
    "User-Agent": "OSS-Guard-Academic-Research/1.0"
})

reintentos = Retry(
    total=4,
    connect=4,
    read=4,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=frozenset(["GET", "POST"]),
    respect_retry_after_header=True,
)

adaptador = HTTPAdapter(max_retries=reintentos)
sesion.mount("https://", adaptador)


# ============================================================
# VERSIONES ESTABLES
# ============================================================

def clave_version(version):
    """Acepta versiones estables con formato mayor.menor.parche."""

    coincidencia = re.fullmatch(
        r"v?(\d+)\.(\d+)\.(\d+)(?:\+.*)?",
        str(version).strip()
    )

    if not coincidencia:
        return None

    return tuple(int(x) for x in coincidencia.groups())


def obtener_versiones_npm(paquete):
    """Consulta npm Registry y devuelve versiones estables ordenadas."""

    nombre_url = quote(paquete, safe="@/")
    url = f"{URL_NPM}/{nombre_url}"

    respuesta = sesion.get(url, timeout=TIMEOUT)
    respuesta.raise_for_status()

    metadata = respuesta.json()
    versiones = metadata.get("versions", {})

    if not isinstance(versiones, dict):
        raise ValueError("Respuesta de npm sin objeto de versiones válido")

    validas = []

    for version in versiones:
        clave = clave_version(version)

        if clave is not None:
            validas.append((clave, version))

    validas.sort(key=lambda elemento: elemento[0])

    return [version for _, version in validas]


def seleccionar_versiones(versiones, limite):
    """
    Selecciona versiones distribuidas a lo largo del historial.
    El mismo criterio se aplica a todos los paquetes.
    """

    if limite <= 0 or not versiones:
        return []

    if len(versiones) <= limite:
        return versiones

    if limite == 1:
        return [versiones[-1]]

    indices = [
        round(i * (len(versiones) - 1) / (limite - 1))
        for i in range(limite)
    ]

    return list(dict.fromkeys(versiones[i] for i in indices))


# ============================================================
# CONSULTAR OSV
# ============================================================

def consultar_osv(paquete, version):
    """
    Consulta vulnerabilidades conocidas para una versión concreta.

    Devuelve el número de vulnerabilidades únicamente si la
    respuesta de OSV es válida. Los errores se propagan al proceso
    principal para que se registren como errores, no como riesgo 0.
    """

    payload = {
        "package": {
            "name": paquete,
            "ecosystem": "npm"
        },
        "version": version
    }

    respuesta = sesion.post(
        URL_OSV,
        json=payload,
        timeout=TIMEOUT
    )

    respuesta.raise_for_status()

    resultado = respuesta.json()
    vulnerabilidades = resultado.get("vulns", [])

    if not isinstance(vulnerabilidades, list):
        raise ValueError("Respuesta de OSV con campo vulns inválido")

    return len(vulnerabilidades)


# ============================================================
# PROCESO PRINCIPAL
# ============================================================

def main():

    if not os.path.isfile(ARCHIVO_ENTRADA):
        raise FileNotFoundError(
            f"No existe {ARCHIVO_ENTRADA}. "
            "Crea ese CSV con una columna llamada 'paquete'."
        )

    entrada = pd.read_csv(ARCHIVO_ENTRADA)

    if "paquete" not in entrada.columns:
        raise ValueError(
            "El CSV de entrada debe tener una columna 'paquete'."
        )

    paquetes = (
        entrada["paquete"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    paquetes = sorted(set(p for p in paquetes if p))

    registros = []
    diagnosticos = []

    print("=" * 65)
    print("OSS GUARD - GENERACIÓN DEL DATASET CON OSV")
    print("=" * 65)
    print(f"Paquetes por procesar: {len(paquetes)}")

    for numero_paquete, paquete in enumerate(paquetes, start=1):

        print(
            f"\n[{numero_paquete}/{len(paquetes)}] {paquete}",
            flush=True
        )

        try:
            versiones = obtener_versiones_npm(paquete)
            seleccionadas = seleccionar_versiones(
                versiones,
                LIMITE_VERSIONES_POR_PAQUETE
            )

        except (requests.RequestException, ValueError) as error:
            diagnosticos.append({
                "paquete": paquete,
                "version": "",
                "estado": "error_npm",
                "detalle": str(error)
            })
            print(f"  Error consultando npm: {error}")
            continue

        print(f"  Versiones seleccionadas: {len(seleccionadas)}")

        for numero_version, version in enumerate(
            seleccionadas,
            start=1
        ):

            print(
                f"  [{numero_version}/{len(seleccionadas)}] "
                f"{paquete}@{version}",
                flush=True
            )

            try:
                cantidad = consultar_osv(paquete, version)

                # La etiqueta se obtiene de la misma respuesta de OSV.
                riesgo = int(cantidad > 0)

                registros.append({
                    "paquete": paquete,
                    "version": version,
                    "numero_vulnerabilidades": cantidad,
                    "riesgo": riesgo
                })

                diagnosticos.append({
                    "paquete": paquete,
                    "version": version,
                    "estado": "ok",
                    "detalle": ""
                })

            except (
                requests.RequestException,
                ValueError
            ) as error:

                # Importante: no crear una etiqueta cuando hubo error.
                diagnosticos.append({
                    "paquete": paquete,
                    "version": version,
                    "estado": "error_osv",
                    "detalle": str(error)
                })

                print(f"    Error OSV: {error}")

            time.sleep(PAUSA_OSV + random.uniform(0, 0.03))

    # Guardar el diagnóstico aunque existan errores.
    pd.DataFrame(
        diagnosticos,
        columns=["paquete", "version", "estado", "detalle"]
    ).to_csv(
        ARCHIVO_DIAGNOSTICO,
        index=False,
        encoding="utf-8"
    )

    if not registros:
        raise RuntimeError(
            "No se obtuvo ninguna respuesta válida de OSV. "
            "No se generó el dataset etiquetado."
        )

    dataset = pd.DataFrame(registros)

    # Validaciones antes de guardar.
    if dataset.duplicated(["paquete", "version"]).any():
        raise RuntimeError(
            "Se encontraron duplicados de paquete y versión."
        )

    inconsistencia = (
        ((dataset["riesgo"] == 0) &
         (dataset["numero_vulnerabilidades"] != 0))
        |
        ((dataset["riesgo"] == 1) &
         (dataset["numero_vulnerabilidades"] < 1))
    )

    if inconsistencia.any():
        raise RuntimeError(
            "Hay etiquetas que no coinciden con el conteo de OSV."
        )

    # Guardar en un temporal y reemplazar solo si se puede leer.
    temporal = ARCHIVO_SALIDA + ".tmp"

    dataset.to_csv(
        temporal,
        index=False,
        encoding="utf-8"
    )

    comprobacion = pd.read_csv(temporal)

    if len(comprobacion) != len(dataset):
        os.remove(temporal)
        raise RuntimeError(
            "La verificación del CSV falló; no se reemplazó el archivo."
        )

    os.replace(temporal, ARCHIVO_SALIDA)

    print("\n" + "=" * 65)
    print("GENERACIÓN FINALIZADA")
    print("=" * 65)
    print(f"Dataset: {ARCHIVO_SALIDA}")
    print(f"Registros etiquetados: {len(dataset)}")
    print(f"Paquetes únicos: {dataset['paquete'].nunique()}")
    print(
        "Distribución de riesgo:",
        dataset["riesgo"].value_counts().sort_index().to_dict()
    )
    print(f"Diagnóstico: {ARCHIVO_DIAGNOSTICO}")
    print(
        "\nLas etiquetas representan las vulnerabilidades conocidas "
        "por OSV al momento de ejecutar la consulta."
    )


if __name__ == "__main__":
    main()
