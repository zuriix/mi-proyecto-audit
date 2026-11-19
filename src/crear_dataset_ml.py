
import os
import time
from datetime import datetime, timezone
from urllib.parse import quote

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ==========================================
# ARCHIVOS Y CONFIGURACIÓN
# ==========================================

ARCHIVO_ENTRADA = "data/raw/dataset_etiquetado_osv.csv"
ARCHIVO_SALIDA = "data/processed/dataset_ml.csv"
ARCHIVO_TEMPORAL = "data/processed/dataset_ml_temporal.csv"
ARCHIVO_DIAGNOSTICO = "data/processed/diagnostico_dataset_ml.csv"

URL_NPM = "https://registry.npmjs.org"

COLUMNAS_SALIDA = [
    "paquete",
    "version",
    "edad_version_dias",
    "versiones_anteriores",
    "numero_dependencias",
    "numero_dependencias_opcionales",
    "numero_scripts",
    "tiene_repositorio",
    "tiene_licencia",
    "riesgo",
]

os.makedirs("data/processed", exist_ok=True)


# ==========================================
# SESIÓN HTTP CON REINTENTOS
# ==========================================

sesion = requests.Session()

reintentos = Retry(
    total=3,
    connect=3,
    read=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
)

adaptador = HTTPAdapter(max_retries=reintentos)
sesion.mount("https://", adaptador)

cache = {}


# ==========================================
# CONSULTAR METADATOS DE NPM
# ==========================================

def obtener_paquete_npm(nombre_paquete):
    if nombre_paquete in cache:
        return cache[nombre_paquete]

    nombre_url = quote(str(nombre_paquete), safe="@/")
    url = f"{URL_NPM}/{nombre_url}"

    try:
        respuesta = sesion.get(url, timeout=30)

        if respuesta.status_code != 200:
            print(
                f"ERROR NPM {respuesta.status_code}: "
                f"{nombre_paquete}"
            )
            return None

        datos = respuesta.json()
        cache[nombre_paquete] = datos
        return datos

    except (requests.RequestException, ValueError) as error:
        print(f"ERROR consultando {nombre_paquete}: {error}")
        return None


# ==========================================
# OBTENER CARACTERÍSTICAS DE UNA VERSIÓN
# ==========================================

def crear_registro(fila, datos_npm):
    paquete = str(fila["paquete"])
    version = str(fila["version"])
    riesgo = int(fila["riesgo"])

    versiones = datos_npm.get("versions", {})
    informacion = versiones.get(version)

    if not isinstance(informacion, dict):
        return None, "version_no_encontrada"

    fechas = datos_npm.get("time", {})
    fecha_version = fechas.get(version)
    edad_version_dias = None
    versiones_anteriores = None

    if fecha_version:
        try:
            fecha = datetime.fromisoformat(
                fecha_version.replace("Z", "+00:00")
            )

            if fecha.tzinfo is None:
                fecha = fecha.replace(tzinfo=timezone.utc)

            edad_version_dias = (
                datetime.now(timezone.utc) - fecha
            ).days

            versiones_anteriores = 0

            for otra_version, otra_fecha in fechas.items():
                if otra_version in ("created", "modified"):
                    continue

                try:
                    fecha_otra = datetime.fromisoformat(
                        otra_fecha.replace("Z", "+00:00")
                    )

                    if fecha_otra.tzinfo is None:
                        fecha_otra = fecha_otra.replace(
                            tzinfo=timezone.utc
                        )

                    # No contar la propia versión.
                    if fecha_otra < fecha:
                        versiones_anteriores += 1

                except (TypeError, ValueError, AttributeError):
                    continue

        except (TypeError, ValueError, AttributeError):
            pass

    dependencias = informacion.get("dependencies") or {}
    opcionales = informacion.get("optionalDependencies") or {}
    scripts = informacion.get("scripts") or {}

    if not isinstance(dependencias, dict):
        dependencias = {}

    if not isinstance(opcionales, dict):
        opcionales = {}

    if not isinstance(scripts, dict):
        scripts = {}

    repositorio = informacion.get("repository")
    licencia = informacion.get("license")

    registro = {
        "paquete": paquete,
        "version": version,
        "edad_version_dias": edad_version_dias,
        "versiones_anteriores": versiones_anteriores,
        "numero_dependencias": len(dependencias),
        "numero_dependencias_opcionales": len(opcionales),
        "numero_scripts": len(scripts),
        "tiene_repositorio": int(bool(repositorio)),
        "tiene_licencia": int(bool(licencia)),
        "riesgo": riesgo,
    }

    return registro, None


# ==========================================
# PROCESO PRINCIPAL
# ==========================================

def main():
    print("=" * 55)
    print("CREANDO DATASET PARA MACHINE LEARNING")
    print("=" * 55)

    if not os.path.exists(ARCHIVO_ENTRADA):
        raise SystemExit(
            f"No existe el archivo de entrada: {ARCHIVO_ENTRADA}"
        )

    dataset = pd.read_csv(ARCHIVO_ENTRADA)

    columnas_requeridas = {
        "paquete",
        "version",
        "riesgo",
    }

    if not columnas_requeridas.issubset(dataset.columns):
        raise SystemExit(
            "El dataset de entrada no contiene las columnas necesarias."
        )

    dataset = dataset.drop_duplicates(
        subset=["paquete", "version"]
    ).copy()

    dataset["riesgo"] = pd.to_numeric(
        dataset["riesgo"], errors="coerce"
    )

    dataset = dataset.dropna(
        subset=["paquete", "version", "riesgo"]
    )

    dataset = dataset[
        dataset["riesgo"].isin([0, 1])
    ].copy()

    dataset["riesgo"] = dataset["riesgo"].astype(int)

    if dataset.empty or set(dataset["riesgo"].unique()) != {0, 1}:
        raise SystemExit(
            "El archivo de entrada debe tener registros de riesgo 0 y 1."
        )

    print(f"Registros de entrada: {len(dataset)}")
    print(f"Paquetes únicos: {dataset['paquete'].nunique()}")

    registros = []
    diagnosticos = []

    total = len(dataset)

    for numero, (_, fila) in enumerate(dataset.iterrows(), start=1):
        paquete = str(fila["paquete"])
        version = str(fila["version"])

        print(
            f"[{numero}/{total}] {paquete}@{version}",
            flush=True,
        )

        datos_npm = obtener_paquete_npm(paquete)

        if datos_npm is None:
            diagnosticos.append({
                "paquete": paquete,
                "version": version,
                "riesgo": int(fila["riesgo"]),
                "estado": "error_npm",
            })
            continue

        registro, error = crear_registro(fila, datos_npm)

        if error:
            diagnosticos.append({
                "paquete": paquete,
                "version": version,
                "riesgo": int(fila["riesgo"]),
                "estado": error,
            })
            continue

        registros.append(registro)

        # Las consultas se hacen una vez por paquete gracias a la caché.
        time.sleep(0.05)

    # Guardar el diagnóstico incluso si hay registros descartados.
    pd.DataFrame(
        diagnosticos,
        columns=["paquete", "version", "riesgo", "estado"],
    ).to_csv(
        ARCHIVO_DIAGNOSTICO,
        index=False,
        encoding="utf-8",
    )

    if not registros:
        raise SystemExit(
            "No se pudo construir ningún registro. "
            "No se modificó el dataset anterior."
        )

    dataset_ml = pd.DataFrame(
        registros,
        columns=COLUMNAS_SALIDA,
    )

    dataset_ml = dataset_ml.drop_duplicates(
        subset=["paquete", "version"]
    )

    # Las fechas faltantes se imputan con la mediana para no
    # descartar automáticamente versiones por falta de metadatos.
    for columna in ["edad_version_dias", "versiones_anteriores"]:
        dataset_ml[columna] = pd.to_numeric(
            dataset_ml[columna], errors="coerce"
        )

        mediana = dataset_ml[columna].median()

        if pd.isna(mediana):
            mediana = 0

        dataset_ml[columna] = dataset_ml[columna].fillna(
            mediana
        )

    dataset_ml = dataset_ml.dropna()

    # ======================================
    # VALIDAR ANTES DE GUARDAR
    # ======================================

    minimo_registros = int(len(dataset) * 0.80)

    if len(dataset_ml) < minimo_registros:
        raise SystemExit(
            f"ERROR: Solo se generaron {len(dataset_ml)} "
            f"de {len(dataset)} registros. "
            "El resultado es demasiado incompleto. "
            "No se reemplazará el dataset anterior."
        )

    if set(dataset_ml["riesgo"].unique()) != {0, 1}:
        raise SystemExit(
            "ERROR: El resultado debe contener las etiquetas 0 y 1. "
            "No se reemplazará el dataset anterior."
        )

    if dataset_ml[COLUMNAS_SALIDA].isna().any().any():
        raise SystemExit(
            "ERROR: Quedan valores nulos. "
            "No se reemplazará el dataset anterior."
        )

    # Escribir primero un archivo temporal.
    dataset_ml.to_csv(
        ARCHIVO_TEMPORAL,
        index=False,
        encoding="utf-8",
    )

    # Comprobar que el temporal se puede leer correctamente.
    comprobacion = pd.read_csv(ARCHIVO_TEMPORAL)

    if len(comprobacion) != len(dataset_ml):
        raise SystemExit(
            "ERROR: La comprobación del archivo temporal falló. "
            "Se conserva el dataset anterior."
        )

    # Reemplazar el archivo final solo después de las validaciones.
    os.replace(ARCHIVO_TEMPORAL, ARCHIVO_SALIDA)

    print()
    print("=" * 55)
    print("DATASET PARA MACHINE LEARNING GENERADO")
    print("=" * 55)
    print(f"Registros de entrada: {len(dataset)}")
    print(f"Registros generados: {len(dataset_ml)}")
    print(f"Paquetes únicos: {dataset_ml['paquete'].nunique()}")
    print("Distribución de riesgo:")
    print(dataset_ml["riesgo"].value_counts().to_dict())
    print(f"Registros con incidencias: {len(diagnosticos)}")
    print(f"Dataset: {ARCHIVO_SALIDA}")
    print(f"Diagnóstico: {ARCHIVO_DIAGNOSTICO}")
    print("=" * 55)


if __name__ == "__main__":
    main()
