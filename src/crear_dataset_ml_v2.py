import os
import re
import time as time_module
from datetime import datetime, timezone

import pandas as pd
import requests


# ============================================================
# CONFIGURACIÓN
# ============================================================

INPUT_FILE = "data/processed/dataset_final.csv"
OUTPUT_FILE = "data/processed/dataset_ml_v2.csv"

NPM_REGISTRY = "https://registry.npmjs.org"

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

session = requests.Session()
session.headers.update({
    "User-Agent": "detector-riesgos-oss/1.0"
})


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def obtener_metadata_npm(paquete):
    """
    Obtiene la metadata completa del paquete desde npm Registry.
    """

    url = f"{NPM_REGISTRY}/{paquete}"

    try:
        respuesta = session.get(url, timeout=30)

        if respuesta.status_code != 200:
            print(
                f"[AVISO] No se pudo obtener {paquete} "
                f"(HTTP {respuesta.status_code})"
            )
            return None

        return respuesta.json()

    except requests.RequestException as e:
        print(f"[ERROR] {paquete}: {e}")
        return None


def convertir_fecha(fecha):
    """
    Convierte una fecha ISO de npm a datetime UTC.
    """

    if not fecha:
        return None

    try:
        fecha = fecha.replace("Z", "+00:00")
        dt = datetime.fromisoformat(fecha)

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        return dt.astimezone(timezone.utc)

    except Exception:
        return None


def extraer_semver(version):
    """
    Extrae mayor, menor y parche de una versión npm.

    Ejemplos:
    4.17.21       -> 4, 17, 21
    5.0.0-beta.1  -> 5, 0, 0
    v1.2.3        -> 1, 2, 3
    """

    if not isinstance(version, str):
        return None, None, None, 0

    patron = r"^v?(\d+)\.(\d+)\.(\d+)(?:[-+].*)?$"

    match = re.match(patron, version.strip())

    if not match:
        return None, None, None, 0

    mayor = int(match.group(1))
    menor = int(match.group(2))
    parche = int(match.group(3))

    es_prerelease = int("-" in version)

    return mayor, menor, parche, es_prerelease


def es_valor_valido(valor):
    """
    Determina si un campo de metadata realmente contiene información.
    """

    if valor is None:
        return False

    if isinstance(valor, str):
        return valor.strip() != ""

    if isinstance(valor, (list, dict)):
        return len(valor) > 0

    return True


def contar_diccionario(valor):
    """
    Cuenta elementos de un diccionario.
    """

    if isinstance(valor, dict):
        return len(valor)

    return 0


def contar_lista(valor):
    """
    Cuenta elementos de una lista.
    """

    if isinstance(valor, list):
        return len(valor)

    return 0


# ============================================================
# CARGAR DATASET ORIGINAL
# ============================================================

print("=" * 70)
print("CREACIÓN DE DATASET ML V2")
print("=" * 70)

print(f"\nLeyendo: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

print(f"Registros encontrados: {len(df)}")


columnas_necesarias = [
    "paquete",
    "version",
    "riesgo"
]

for columna in columnas_necesarias:
    if columna not in df.columns:
        raise ValueError(
            f"No existe la columna requerida: {columna}"
        )


# ============================================================
# CACHE PARA NO CONSULTAR npm REPETIDAMENTE
# ============================================================

cache = {}

resultados = []


# ============================================================
# PROCESAMIENTO
# ============================================================

for indice, fila in df.iterrows():

    paquete = str(fila["paquete"]).strip()
    version = str(fila["version"]).strip()
    riesgo = int(fila["riesgo"])

    print(
        f"\n[{indice + 1}/{len(df)}] "
        f"{paquete}@{version}"
    )

    # --------------------------------------------------------
    # Obtener metadata del paquete
    # --------------------------------------------------------

    if paquete not in cache:

        metadata = obtener_metadata_npm(paquete)

        if metadata is None:
            print(
                f"[AVISO] Se omite {paquete}@{version}"
            )
            continue

        cache[paquete] = metadata

        # Evitar saturar npm Registry
        time_module.sleep(0.1)

    else:
        metadata = cache[paquete]


    # --------------------------------------------------------
    # Obtener metadata específica de la versión
    # --------------------------------------------------------

    versiones = metadata.get("versions", {})

    version_data = versiones.get(version)

    if version_data is None:

        print(
            f"[AVISO] No existe metadata específica "
            f"para {paquete}@{version}"
        )

        continue


    # ========================================================
    # FECHA DE PUBLICACIÓN
    # ========================================================

    tiempos = metadata.get("time", {})

    fecha_publicacion_texto = tiempos.get(version)

    fecha_publicacion = convertir_fecha(
        fecha_publicacion_texto
    )

    if fecha_publicacion is None:

        print(
            f"[AVISO] No existe fecha de publicación "
            f"para {paquete}@{version}"
        )

        continue


    # ========================================================
    # VERSION SEMÁNTICA
    # ========================================================

    (
        version_mayor,
        version_menor,
        version_parche,
        es_prerelease
    ) = extraer_semver(version)

    if version_mayor is None:

        print(
            f"[AVISO] Versión no compatible con SemVer: "
            f"{paquete}@{version}"
        )

        continue


    # ========================================================
    # VERSIÓN ANTERIOR
    # ========================================================

    fechas_versiones = []

    for version_npm, fecha_npm in tiempos.items():

        if version_npm == "created":
            continue

        if version_npm == "modified":
            continue

        fecha = convertir_fecha(fecha_npm)

        if fecha is None:
            continue

        fechas_versiones.append(
            (version_npm, fecha)
        )


    # Orden cronológico
    fechas_versiones.sort(
        key=lambda x: x[1]
    )


    # Buscar versiones anteriores a la actual
    versiones_anteriores = [
        (v, f)
        for v, f in fechas_versiones
        if f < fecha_publicacion
    ]


    numero_versiones_anteriores = len(
        versiones_anteriores
    )


    # --------------------------------------------------------
    # Días desde la versión anterior
    # --------------------------------------------------------

    if versiones_anteriores:

        fecha_anterior = versiones_anteriores[-1][1]

        dias_desde_version_anterior = (
            fecha_publicacion - fecha_anterior
        ).total_seconds() / 86400

    else:

        dias_desde_version_anterior = 0


    # ========================================================
    # DEPENDENCIAS
    # ========================================================

    dependencies = version_data.get(
        "dependencies",
        {}
    )

    optional_dependencies = version_data.get(
        "optionalDependencies",
        {}
    )

    dev_dependencies = version_data.get(
        "devDependencies",
        {}
    )

    peer_dependencies = version_data.get(
        "peerDependencies",
        {}
    )

    engines = version_data.get(
        "engines",
        {}
    )

    scripts = version_data.get(
        "scripts",
        {}
    )

    files = version_data.get(
        "files",
        []
    )

    keywords = version_data.get(
        "keywords",
        []
    )


    # ========================================================
    # METADATA DE LA VERSIÓN
    # ========================================================

    repository = version_data.get(
        "repository"
    )

    license = version_data.get(
        "license"
    )

    homepage = version_data.get(
        "homepage"
    )

    readme = version_data.get(
        "readme"
    )


    # ========================================================
    # CREAR REGISTRO
    # ========================================================

    registro = {

        # Identificación
        "paquete": paquete,
        "version": version,

        # Fecha: se conserva para realizar posteriormente
        # la división temporal. NO será entrada del modelo.
        "fecha_publicacion": fecha_publicacion.isoformat(),

        # Características de versión
        "version_mayor": version_mayor,
        "version_menor": version_menor,
        "version_parche": version_parche,
        "es_prerelease": es_prerelease,

        # Historial disponible hasta esa versión
        "versiones_anteriores": numero_versiones_anteriores,
        "dias_desde_version_anterior": round(
            dias_desde_version_anterior,
            2
        ),

        # Dependencias
        "numero_dependencias": contar_diccionario(
            dependencies
        ),

        "numero_dependencias_opcionales": contar_diccionario(
            optional_dependencies
        ),

        "numero_dev_dependencies": contar_diccionario(
            dev_dependencies
        ),

        "numero_peer_dependencies": contar_diccionario(
            peer_dependencies
        ),

        # Otros elementos del package.json
        "numero_scripts": contar_diccionario(
            scripts
        ),

        "numero_engines": contar_diccionario(
            engines
        ),

        "numero_files": contar_lista(
            files
        ),

        "numero_keywords": contar_lista(
            keywords
        ),

        # Metadata
        "tiene_repositorio": int(
            es_valor_valido(repository)
        ),

        "tiene_licencia": int(
            es_valor_valido(license)
        ),

        "tiene_homepage": int(
            es_valor_valido(homepage)
        ),

        "tiene_readme": int(
            es_valor_valido(readme)
        ),

        # Variable objetivo
        "riesgo": riesgo
    }


    resultados.append(registro)


# ============================================================
# CREAR DATAFRAME
# ============================================================

df_ml = pd.DataFrame(resultados)


# ============================================================
# LIMPIEZA
# ============================================================

print("\n" + "=" * 70)
print("LIMPIEZA")
print("=" * 70)

print(
    f"Registros antes de limpieza: {len(df_ml)}"
)


# Eliminar duplicados
df_ml = df_ml.drop_duplicates(
    subset=["paquete", "version"]
)


# Eliminar registros sin target
df_ml = df_ml.dropna(
    subset=["riesgo"]
)


# Asegurar que riesgo sea entero
df_ml["riesgo"] = (
    pd.to_numeric(
        df_ml["riesgo"],
        errors="coerce"
    )
)


df_ml = df_ml[
    df_ml["riesgo"].isin([0, 1])
]


df_ml["riesgo"] = df_ml["riesgo"].astype(int)


# ============================================================
# ORDENAR
# ============================================================

df_ml = df_ml.sort_values(
    by=[
        "paquete",
        "fecha_publicacion"
    ]
).reset_index(drop=True)


# ============================================================
# GUARDAR
# ============================================================

df_ml.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)


# ============================================================
# RESUMEN
# ============================================================

print("\n" + "=" * 70)
print("DATASET ML V2 GENERADO")
print("=" * 70)

print(
    f"Registros finales: {len(df_ml)}"
)

print(
    f"Columnas: {len(df_ml.columns)}"
)

print("\nColumnas:")

for columna in df_ml.columns:
    print(f" - {columna}")


print("\nDistribución de riesgo:")

print(
    df_ml["riesgo"]
    .value_counts()
    .sort_index()
)


print("\nPorcentajes:")

print(
    (
        df_ml["riesgo"]
        .value_counts(normalize=True)
        .sort_index() * 100
    ).round(2)
)


print("\nValores nulos:")

print(
    df_ml.isnull().sum()
)


print("\nPrimeros registros:")

print(
    df_ml.head(10).to_string(
        index=False
    )
)


print("\nArchivo generado:")

print(
    OUTPUT_FILE
)

print("\nProceso terminado correctamente.")