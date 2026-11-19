
import os
import time

import pandas as pd
import requests


# ============================================================
# CONFIGURACIÓN
# ============================================================

ARCHIVO_ENTRADA = "data/raw/dataset_npm.csv"
ARCHIVO_SALIDA = (
    "data/raw/dataset_versiones_vulnerables_v2.csv"
)

LIMITE_VERSIONES_POR_PAQUETE = 100

URL_NPM = "https://registry.npmjs.org"
URL_OSV = "https://api.osv.dev/v1/query"


# ============================================================
# CONSULTAR VERSIONES DISPONIBLES EN NPM
# ============================================================

def obtener_versiones(paquete):
    url = f"{URL_NPM}/{paquete}"

    try:
        respuesta = requests.get(url, timeout=30)
        respuesta.raise_for_status()

        datos = respuesta.json()
        versiones = list(datos.get("versions", {}).keys())

        return versiones

    except (requests.RequestException, ValueError) as error:
        print(
            f"Error consultando NPM para {paquete}: {error}"
        )
        return None


# ============================================================
# VALIDAR VERSIONES
# ============================================================

def es_version_valida(version):
    # Excluir versiones prerelease: beta, alpha, rc, etc.
    if "-" in version:
        return False

    return True


# ============================================================
# SELECCIONAR VERSIONES DISTRIBUIDAS EN EL HISTORIAL
# ============================================================

def seleccionar_versiones(versiones, limite=100):
    versiones_validas = [
        version
        for version in versiones
        if es_version_valida(version)
    ]

    if not versiones_validas:
        return []

    if len(versiones_validas) <= limite:
        return versiones_validas

    # Distribuir la selección a lo largo de la lista.
    # Nota: el orden de NPM no garantiza orden semántico.
    posiciones = []

    for i in range(limite):
        posicion = round(
            i * (len(versiones_validas) - 1) / (limite - 1)
        )
        posiciones.append(posicion)

    seleccionadas = [
        versiones_validas[posicion]
        for posicion in posiciones
    ]

    return list(dict.fromkeys(seleccionadas))


# ============================================================
# CONSULTAR VULNERABILIDADES EN OSV
# ============================================================

def consultar_osv(paquete, version):
    payload = {
        "package": {
            "name": paquete,
            "ecosystem": "npm"
        },
        "version": version
    }

    try:
        respuesta = requests.post(
            URL_OSV,
            json=payload,
            timeout=30
        )

        respuesta.raise_for_status()
        datos = respuesta.json()

        vulnerabilidades = datos.get("vulns", [])

        return len(vulnerabilidades)

    except (requests.RequestException, ValueError) as error:
        print(
            f"Error consultando OSV para "
            f"{paquete} {version}: {error}"
        )
        return None


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():
    print("=" * 70)
    print("GENERACIÓN DE VERSIONES VULNERABLES")
    print("=" * 70)

    # Verificar archivo de entrada
    if not os.path.exists(ARCHIVO_ENTRADA):
        print(
            f"\nNo se encontró el archivo:\n"
            f"{ARCHIVO_ENTRADA}"
        )
        return

    try:
        df = pd.read_csv(ARCHIVO_ENTRADA)
    except (OSError, pd.errors.ParserError) as error:
        print(f"Error leyendo el CSV: {error}")
        return

    # Verificar columna necesaria
    if "paquete" not in df.columns:
        print(
            "Error: el archivo de entrada debe tener "
            "una columna llamada 'paquete'."
        )
        return

    paquetes = (
        df["paquete"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if not paquetes:
        print("No se encontraron paquetes para analizar.")
        return

    print(f"\nPaquetes encontrados: {len(paquetes)}")
    print(
        "Máximo de versiones por paquete: "
        f"{LIMITE_VERSIONES_POR_PAQUETE}"
    )

    registros = []
    consultas_fallidas = 0

    # Procesar cada paquete
    for numero, paquete in enumerate(paquetes, start=1):
        print(
            f"\n[{numero}/{len(paquetes)}] "
            f"Procesando: {paquete}"
        )

        versiones = obtener_versiones(paquete)

        # Diferenciar error de consulta de lista vacía
        if versiones is None:
            print(
                f"No se pudo obtener el historial de {paquete}. "
                "Se omite el paquete."
            )
            consultas_fallidas += 1
            continue

        print(f"Versiones encontradas: {len(versiones)}")

        versiones_seleccionadas = seleccionar_versiones(
            versiones,
            LIMITE_VERSIONES_POR_PAQUETE
        )

        print(
            "Versiones seleccionadas: "
            f"{len(versiones_seleccionadas)}"
        )

        vulnerables_paquete = 0

        # Consultar cada versión en OSV
        for indice, version in enumerate(
            versiones_seleccionadas,
            start=1
        ):
            numero_vulnerabilidades = consultar_osv(
                paquete,
                version
            )

            # Si la consulta falla, no clasificar la versión
            if numero_vulnerabilidades is None:
                consultas_fallidas += 1
                print(
                    f"  Consulta fallida: {paquete} {version}. "
                    "Se omite."
                )
                continue

            # Guardar únicamente versiones con vulnerabilidades
            if numero_vulnerabilidades > 0:
                registros.append({
                    "paquete": paquete,
                    "version": version,
                    "numero_vulnerabilidades":
                        numero_vulnerabilidades,
                    "riesgo": 1
                })

                vulnerables_paquete += 1

                print(
                    f"  [{indice}/"
                    f"{len(versiones_seleccionadas)}] "
                    f"{version}: "
                    f"{numero_vulnerabilidades} "
                    "vulnerabilidad(es)"
                )

            elif indice % 10 == 0:
                print(
                    f"  {indice}/"
                    f"{len(versiones_seleccionadas)} "
                    "versiones revisadas..."
                )

            # Pausa entre consultas
            time.sleep(0.05)

        print(
            f"Versiones vulnerables encontradas para "
            f"{paquete}: {vulnerables_paquete}"
        )

    # Crear DataFrame con columnas incluso si no hay resultados
    columnas = [
        "paquete",
        "version",
        "numero_vulnerabilidades",
        "riesgo"
    ]

    resultado = pd.DataFrame(registros, columns=columnas)

    if resultado.empty:
        print(
            "\nNo se encontraron versiones vulnerables "
            "en las consultas completadas."
        )
        print(
            "No se ha sobrescrito el archivo de salida."
        )
        print(f"Consultas fallidas: {consultas_fallidas}")
        return

    # Eliminar duplicados
    resultado = resultado.drop_duplicates(
        subset=["paquete", "version"]
    )

    # Ordenar por paquete y versión (orden alfabético)
    resultado = resultado.sort_values(
        by=["paquete", "version"]
    ).reset_index(drop=True)

    # Crear carpeta de salida
    carpeta_salida = os.path.dirname(ARCHIVO_SALIDA)
    if carpeta_salida:
        os.makedirs(carpeta_salida, exist_ok=True)

    # Guardar archivo
    resultado.to_csv(
        ARCHIVO_SALIDA,
        index=False,
        encoding="utf-8-sig"
    )

    # Resumen final
    print("\n" + "=" * 70)
    print("RESULTADO FINAL")
    print("=" * 70)

    print(f"\nTotal de registros vulnerables: {len(resultado)}")

    print("\nRegistros por paquete:")
    print(
        resultado["paquete"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nDistribución de riesgo:")
    print(
        resultado["riesgo"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(f"\nConsultas fallidas: {consultas_fallidas}")
    print(f"\nArchivo generado: {ARCHIVO_SALIDA}")

    print("\n" + "=" * 70)
    print("PROCESO TERMINADO")
    print("=" * 70)


if __name__ == "__main__":
    main()