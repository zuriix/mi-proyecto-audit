import os
import time
import random
import requests
import pandas as pd

# ============================================================
# CONFIGURACION
# ============================================================
ARCHIVO_ENTRADA = "data/raw/dataset_npm.csv"
ARCHIVO_SALIDA = "data/raw/dataset_versiones_no_vulnerables.csv"
ARCHIVO_DIAGNOSTICO = "data/raw/diagnostico_no_vulnerables.csv"

LIMITE_VERSIONES_POR_PAQUETE = 300
URL_NPM = "https://registry.npmjs.org"
URL_OSV = "https://api.osv.dev/v1/query"
PAUSA_ENTRE_PAQUETES = 2
PAUSA_ENTRE_CONSULTAS_OSV = 0.15
MAX_REINTENTOS_NPM = 5
MAX_REINTENTOS_OSV = 3
TIMEOUT_HTTP = 30

session = requests.Session()
session.headers.update({"User-Agent": "OSS-Risk-Detector-Universidad/1.0"})


def consultar_osv(paquete, version):
    """Devuelve el numero de vulnerabilidades, o None si la consulta fallo."""
    datos = {
        "package": {"name": paquete, "ecosystem": "npm"},
        "version": version
    }

    for intento in range(1, MAX_REINTENTOS_OSV + 1):
        try:
            respuesta = session.post(URL_OSV, json=datos, timeout=TIMEOUT_HTTP)

            if respuesta.status_code == 200:
                try:
                    resultado = respuesta.json()
                except ValueError as error:
                    print(f"    JSON invalido de OSV para {paquete}@{version}: {error}")
                    return None

                vulnerabilidades = resultado.get("vulns", [])
                if not isinstance(vulnerabilidades, list):
                    print(f"    Respuesta inesperada de OSV para {paquete}@{version}")
                    return None
                return len(vulnerabilidades)

            if respuesta.status_code == 429 or respuesta.status_code >= 500:
                espera = min(5 * intento, 30) + random.uniform(0, 1)
                print(f"    OSV HTTP {respuesta.status_code} para {paquete}@{version}; espera {espera:.1f}s")
                if intento < MAX_REINTENTOS_OSV:
                    time.sleep(espera)
                    continue
                return None

            print(f"    Error HTTP OSV {respuesta.status_code} para {paquete}@{version}: {respuesta.text[:200]}")
            return None

        except requests.RequestException as error:
            print(f"    Error de conexion OSV para {paquete}@{version}: {error}")
            if intento < MAX_REINTENTOS_OSV:
                time.sleep(3 * intento)
            else:
                return None

    return None


def obtener_versiones(paquete):
    """Devuelve lista de versiones estables o None si npm fallo."""
    url = f"{URL_NPM}/{paquete}"

    for intento in range(1, MAX_REINTENTOS_NPM + 1):
        try:
            respuesta = session.get(url, timeout=TIMEOUT_HTTP)
            print(f"  NPM: {paquete} | HTTP {respuesta.status_code} | Intento {intento}/{MAX_REINTENTOS_NPM}")

            if respuesta.status_code == 200:
                try:
                    datos = respuesta.json()
                except ValueError as error:
                    print(f"  JSON invalido de npm para {paquete}: {error}")
                    return None

                versiones_dict = datos.get("versions", {})
                if not isinstance(versiones_dict, dict):
                    print(f"  Formato inesperado de versiones para {paquete}")
                    return None

                versiones = list(versiones_dict.keys())
                versiones_validas = [v for v in versiones if "-" not in v]
                print(f"  Versiones encontradas: {len(versiones)}")
                print(f"  Versiones estables: {len(versiones_validas)}")
                return versiones_validas

            if respuesta.status_code == 429 or respuesta.status_code >= 500:
                espera = min(5 * intento, 30) + random.uniform(0, 1)
                print(f"  Error temporal npm HTTP {respuesta.status_code}; espera {espera:.1f}s")
                if intento < MAX_REINTENTOS_NPM:
                    time.sleep(espera)
                    continue
                return None

            print(f"  Error HTTP npm {respuesta.status_code} para {paquete}: {respuesta.text[:200]}")
            return None

        except requests.RequestException as error:
            print(f"  Error de conexion npm para {paquete}: {error}")
            if intento < MAX_REINTENTOS_NPM:
                espera = 5 * intento
                print(f"  Reintentando en {espera}s...")
                time.sleep(espera)
            else:
                return None

    return None


def seleccionar_versiones(versiones, limite=LIMITE_VERSIONES_POR_PAQUETE):
    if not versiones or limite <= 0:
        return []
    if len(versiones) <= limite:
        return versiones
    if limite == 1:
        return [versiones[len(versiones) // 2]]

    posiciones = [round(i * (len(versiones) - 1) / (limite - 1)) for i in range(limite)]
    return [versiones[posicion] for posicion in posiciones]


def main():
    print("\n" + "=" * 70)
    print("GENERACION DE VERSIONES NO VULNERABLES Y DIAGNOSTICO")
    print("=" * 70)

    if not os.path.exists(ARCHIVO_ENTRADA):
        raise FileNotFoundError(f"No existe el archivo de entrada: {ARCHIVO_ENTRADA}")

    df_entrada = pd.read_csv(ARCHIVO_ENTRADA)
    if "paquete" not in df_entrada.columns:
        raise ValueError(f"El archivo {ARCHIVO_ENTRADA} no contiene la columna 'paquete'.")

    paquetes = df_entrada["paquete"].dropna().astype(str).str.strip()
    paquetes = paquetes[paquetes != ""].unique().tolist()

    print(f"Paquetes encontrados: {len(paquetes)}")
    print(f"Limite de versiones por paquete: {LIMITE_VERSIONES_POR_PAQUETE}")

    registros_no_vulnerables = []
    diagnostico = []

    for indice, paquete in enumerate(paquetes, start=1):
        print("\n" + "=" * 70)
        print(f"[{indice}/{len(paquetes)}] PAQUETE: {paquete}")
        print("=" * 70)

        versiones = obtener_versiones(paquete)

        if versiones is None:
            diagnostico.append({
                "paquete": paquete,
                "versiones_npm": None,
                "versiones_revisadas": 0,
                "vulnerables": 0,
                "no_vulnerables": 0,
                "errores": 1,
                "estado_npm": "error",
                "detalle": "No se pudieron obtener versiones de npm"
            })
            time.sleep(PAUSA_ENTRE_PAQUETES)
            continue

        if not versiones:
            diagnostico.append({
                "paquete": paquete,
                "versiones_npm": 0,
                "versiones_revisadas": 0,
                "vulnerables": 0,
                "no_vulnerables": 0,
                "errores": 0,
                "estado_npm": "sin_versiones_estables",
                "detalle": "npm respondio correctamente, sin versiones estables"
            })
            time.sleep(PAUSA_ENTRE_PAQUETES)
            continue

        seleccionadas = seleccionar_versiones(versiones)
        contador_vulnerables = 0
        contador_no_vulnerables = 0
        contador_errores = 0

        print(f"Versiones estables disponibles: {len(versiones)}")
        print(f"Versiones seleccionadas para consulta: {len(seleccionadas)}")

        for numero, version in enumerate(seleccionadas, start=1):
            resultado = consultar_osv(paquete, version)

            if resultado is None:
                contador_errores += 1
            elif resultado == 0:
                contador_no_vulnerables += 1
                registros_no_vulnerables.append({
                    "paquete": paquete,
                    "version": version,
                    "numero_vulnerabilidades": 0,
                    "riesgo": 0
                })
            else:
                contador_vulnerables += 1

            if numero % 25 == 0 or numero == len(seleccionadas):
                print(f"  Progreso {numero}/{len(seleccionadas)} | Vulnerables: {contador_vulnerables} | No vulnerables: {contador_no_vulnerables} | Errores: {contador_errores}")

            time.sleep(PAUSA_ENTRE_CONSULTAS_OSV)

        diagnostico.append({
            "paquete": paquete,
            "versiones_npm": len(versiones),
            "versiones_revisadas": len(seleccionadas),
            "vulnerables": contador_vulnerables,
            "no_vulnerables": contador_no_vulnerables,
            "errores": contador_errores,
            "estado_npm": "ok",
            "detalle": ""
        })
        time.sleep(PAUSA_ENTRE_PAQUETES)

    os.makedirs(os.path.dirname(ARCHIVO_SALIDA), exist_ok=True)
    columnas = ["paquete", "version", "numero_vulnerabilidades", "riesgo"]
    df_salida = pd.DataFrame(registros_no_vulnerables, columns=columnas)

    if not df_salida.empty:
        df_salida = df_salida.drop_duplicates(subset=["paquete", "version"])
        df_salida = df_salida.sort_values(["paquete", "version"])

    df_salida.to_csv(ARCHIVO_SALIDA, index=False, encoding="utf-8-sig")

    df_diagnostico = pd.DataFrame(diagnostico)
    df_diagnostico.to_csv(ARCHIVO_DIAGNOSTICO, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print("RESUMEN FINAL")
    print("=" * 70)
    print(f"Paquetes procesados: {len(diagnostico)}")
    print(f"Versiones no vulnerables guardadas: {len(df_salida)}")
    print(f"Total de versiones vulnerables detectadas: {sum(d['vulnerables'] for d in diagnostico)}")
    print(f"Total de errores OSV: {sum(d['errores'] for d in diagnostico)}")
    print(f"Dataset guardado en: {ARCHIVO_SALIDA}")
    print(f"Diagnostico guardado en: {ARCHIVO_DIAGNOSTICO}")

    if not df_salida.empty:
        print("Etiquetas generadas:", df_salida["riesgo"].value_counts(dropna=False).to_dict())
    else:
        print("ADVERTENCIA: no se generaron registros no vulnerables. Revisa el diagnostico.")


if __name__ == "__main__":
    main()
