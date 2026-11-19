
from pathlib import Path

import joblib
import pandas as pd

from detectar_riesgo import (
    RUTA_MODELO,
    obtener_datos_npm,
    auditar_vulnerabilidades,
)


BASE_DIR = Path(__file__).resolve().parent.parent
CARPETA_RESULTADOS = BASE_DIR / "data" / "processed"
ARCHIVO_CSV = CARPETA_RESULTADOS / "resultados_pruebas.csv"


PAQUETES = [
    ("lodash", "4.17.21"),
    ("minimist", "1.2.5"),
    ("express", "4.21.2"),
    ("axios", "1.7.9"),
    ("debug", "4.3.7"),
    ("jquery", "3.7.1"),
    ("react", "18.3.1"),
    ("semver", "7.6.3"),
    ("uuid", "9.0.1"),
    ("moment", "2.30.1"),
]



def probar_paquete(paquete, version, modelo, columnas):
    print(f"\nAnalizando {paquete}@{version}...")

    resultado = {
        "paquete": paquete,
        "version": version,
        "clasificacion_ml": "ERROR",
        "probabilidad_riesgo": None,
        "auditoria": "NO COMPLETADA",
        "paquetes_afectados": None,
        "criticas": None,
        "altas": None,
        "moderadas": None,
        "bajas": None,
        "resultado_general": "NO DETERMINADO",
        "detalle_error": "",
    }

    try:
        caracteristicas = obtener_datos_npm(paquete, version)

        faltantes = set(columnas) - set(caracteristicas)
        if faltantes:
            raise ValueError(
                f"Faltan caracteristicas: {sorted(faltantes)}"
            )

        X = pd.DataFrame([{
            nombre: caracteristicas[nombre]
            for nombre in columnas
        }])

        prediccion = int(modelo.predict(X)[0])
        clases = list(modelo.classes_)
        probabilidades = modelo.predict_proba(X)[0]

        probabilidad = (
            float(probabilidades[clases.index(1)])
            if 1 in clases else 0.0
        )

        resultado["clasificacion_ml"] = (
            "RIESGO" if prediccion == 1
            else "SIN RIESGO DETECTADO"
        )
        resultado["probabilidad_riesgo"] = round(
            probabilidad, 4
        )

    except Exception as error:
        resultado["detalle_error"] = (
            f"Error en modelo/metadatos: {error}"
        )
        print(resultado["detalle_error"])
        return resultado

    try:
        auditoria = auditar_vulnerabilidades(paquete, version)

        if auditoria["estado"] != "completado":
            resultado["auditoria"] = "ERROR"
            resultado["detalle_error"] = auditoria.get(
                "mensaje", "Error desconocido en npm audit"
            )
            resultado["resultado_general"] = (
                "NO DETERMINADO: AUDITORIA FALLIDA"
            )
            print("No se pudo completar npm audit.")
            return resultado

        resultado["auditoria"] = "COMPLETADA"
        resultado["paquetes_afectados"] = auditoria["total"]
        resultado["criticas"] = auditoria["criticas"]
        resultado["altas"] = auditoria["altas"]
        resultado["moderadas"] = auditoria["moderadas"]
        resultado["bajas"] = auditoria["bajas"]

        if auditoria["total"] > 0:
            resultado["resultado_general"] = "RIESGO"
        elif prediccion == 1:
            resultado["resultado_general"] = (
                "RIESGO SEGUN EL MODELO"
            )
        else:
            resultado["resultado_general"] = (
                "SIN RIESGO DETECTADO"
            )

    except Exception as error:
        resultado["auditoria"] = "ERROR"
        resultado["detalle_error"] = (
            f"Error en auditoria: {error}"
        )
        resultado["resultado_general"] = (
            "NO DETERMINADO: AUDITORIA FALLIDA"
        )

    print(
        f"ML: {resultado['clasificacion_ml']} | "
        f"npm audit: {resultado['auditoria']} | "
        f"Resultado: {resultado['resultado_general']}"
    )

    return resultado


def main():
    if not RUTA_MODELO.is_file():
        print(f"No se encontro el modelo: {RUTA_MODELO}")
        return

    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)

    artefacto = joblib.load(RUTA_MODELO)
    modelo = artefacto["modelo"]
    columnas = artefacto["caracteristicas"]

    resultados = []

    for paquete, version in PAQUETES:
        resultado = probar_paquete(
            paquete, version, modelo, columnas
        )
        resultados.append(resultado)

    df = pd.DataFrame(resultados)
    df.to_csv(ARCHIVO_CSV, index=False, encoding="utf-8-sig")

    print("\n=== RESUMEN DE LAS PRUEBAS ===")
    print(df[
        [
            "paquete",
            "version",
            "clasificacion_ml",
            "auditoria",
            "paquetes_afectados",
            "resultado_general",
        ]
    ].to_string(index=False))

    print(f"\nCSV guardado en:\n{ARCHIVO_CSV}")
    print(
        "\nNota: que npm audit no encuentre avisos no "
        "garantiza que el paquete sea seguro."
    )


if __name__ == "__main__":
    main()
