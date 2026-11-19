
from pathlib import Path
from datetime import datetime, timezone
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from urllib.parse import quote
import json
import os
import shutil
import subprocess
import tempfile

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
RUTA_MODELO = BASE_DIR / "models" / "random_forest.pkl"
URL_NPM = "https://registry.npmjs.org"

# Buscar npm en la ubicación habitual de Windows y, después, en PATH.
NPM_CANDIDATOS = [
    Path(r"C:\Program Files\nodejs\npm.cmd"),
    Path(r"C:\Program Files (x86)\nodejs\npm.cmd"),
]

NPM = next(
    (ruta for ruta in NPM_CANDIDATOS if ruta.is_file()),
    None,
)

if NPM is None:
    npm_en_path = shutil.which("npm.cmd") or shutil.which("npm")
    if npm_en_path:
        NPM = Path(npm_en_path)


def obtener_datos_npm(paquete, version):
    """Obtiene las características de una versión publicada en npm."""
    paquete = paquete.strip()
    version = version.strip()

    if not paquete or not version:
        raise ValueError("El paquete y la versión son obligatorios.")

    url = f"{URL_NPM}/{quote(paquete, safe='@')}"
    solicitud = Request(
        url,
        headers={"User-Agent": "DetectorRiesgosOSS/1.0"},
    )

    with urlopen(solicitud, timeout=30) as respuesta:
        metadatos = json.load(respuesta)

    versiones = metadatos.get("versions", {})

    if version not in versiones:
        raise ValueError(
            f"No se encontró la versión {version} "
            f"del paquete {paquete} en npm."
        )

    datos = versiones[version]
    fecha_texto = metadatos.get("time", {}).get(version)

    if not fecha_texto:
        raise ValueError(
            "npm no proporciona la fecha de publicación de esta versión."
        )

    fecha = datetime.fromisoformat(
        fecha_texto.replace("Z", "+00:00")
    )

    ahora = datetime.now(timezone.utc)
    edad = max(
        0,
        (ahora - fecha).total_seconds() / 86400,
    )

    anteriores = 0

    for nombre, texto in metadatos.get("time", {}).items():
        if nombre in ("created", "modified", version):
            continue

        try:
            fecha_otra = datetime.fromisoformat(
                texto.replace("Z", "+00:00")
            )

            if fecha_otra < fecha:
                anteriores += 1

        except (ValueError, AttributeError, TypeError):
            continue

    return {
        "edad_version_dias": edad,
        "versiones_anteriores": anteriores,
        "numero_dependencias": len(
            datos.get("dependencies") or {}
        ),
        "numero_dependencias_opcionales": len(
            datos.get("optionalDependencies") or {}
        ),
        "numero_scripts": len(
            datos.get("scripts") or {}
        ),
        "tiene_repositorio": int(
            bool(datos.get("repository"))
        ),
        "tiene_licencia": int(
            bool(datos.get("license"))
        ),
    }


def ejecutar_npm(argumentos, carpeta, timeout=120):
    """Ejecuta npm de forma compatible con Windows."""
    if NPM is None or not NPM.is_file():
        raise FileNotFoundError(
            "No se encontró npm. Comprueba la instalación de Node.js."
        )

    argumentos = [str(arg) for arg in argumentos]

    if os.name == "nt":
        comando = subprocess.list2cmdline(
            [str(NPM), *argumentos]
        )

        return subprocess.run(
            comando,
            cwd=carpeta,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            shell=True,
        )

    return subprocess.run(
        [str(NPM), *argumentos],
        cwd=carpeta,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def auditar_vulnerabilidades(paquete, version):
    """Ejecuta npm audit en una carpeta temporal independiente."""
    if NPM is None or not NPM.is_file():
        return {
            "estado": "error",
            "mensaje": (
                "No se encontró npm. Verifica que Node.js esté instalado "
                "y que npm esté disponible."
            ),
        }

    with tempfile.TemporaryDirectory(
        prefix="auditoria_oss_"
    ) as carpeta:
        try:
            # Crear un proyecto temporal para generar package-lock.json.
            instalacion = ejecutar_npm(
                [
                    "install",
                    "--package-lock-only",
                    "--ignore-scripts",
                    "--no-audit",
                    "--no-fund",
                    "--save-exact",
                    f"{paquete}@{version}",
                ],
                carpeta,
            )

            if instalacion.returncode != 0:
                detalle = (
                    instalacion.stderr or instalacion.stdout
                ).strip()

                return {
                    "estado": "error",
                    "mensaje": (
                        detalle[-1500:]
                        or "No se pudo preparar la auditoría."
                    ),
                }

            resultado = ejecutar_npm(
                ["audit", "--json"],
                carpeta,
            )

            try:
                informe = json.loads(resultado.stdout)
            except (json.JSONDecodeError, TypeError):
                detalle = (
                    resultado.stderr or resultado.stdout
                ).strip()

                return {
                    "estado": "error",
                    "mensaje": (
                        detalle[-1500:]
                        or "npm no devolvió un informe JSON válido."
                    ),
                }

            if not isinstance(informe, dict):
                return {
                    "estado": "error",
                    "mensaje": "El informe de npm tiene un formato inválido.",
                }

            if not isinstance(informe.get("metadata"), dict):
                error_npm = informe.get("error") or {}

                return {
                    "estado": "error",
                    "mensaje": error_npm.get(
                        "summary",
                        "No se pudo interpretar el informe de npm.",
                    ),
                }

            totales = informe["metadata"].get(
                "vulnerabilities", {}
            )

            return {
                "estado": "completado",
                "total": totales.get("total", 0),
                "criticas": totales.get("critical", 0),
                "altas": totales.get("high", 0),
                "moderadas": totales.get("moderate", 0),
                "bajas": totales.get("low", 0),
                "info": totales.get("info", 0),
                "avisos": informe.get("vulnerabilities", {}),
            }

        except subprocess.TimeoutExpired:
            return {
                "estado": "error",
                "mensaje": (
                    "La auditoría tardó demasiado. "
                    "Inténtalo nuevamente."
                ),
            }

        except (OSError, subprocess.SubprocessError) as error:
            return {
                "estado": "error",
                "mensaje": str(error),
            }


def cargar_modelo():
    """Carga el modelo y la lista de características guardadas."""
    if not RUTA_MODELO.is_file():
        raise FileNotFoundError(
            f"No se encontró el modelo: {RUTA_MODELO}"
        )

    artefacto = joblib.load(RUTA_MODELO)

    if not isinstance(artefacto, dict):
        raise ValueError(
            "El archivo del modelo no contiene el artefacto esperado."
        )

    modelo = artefacto.get("modelo")
    columnas = artefacto.get("caracteristicas")

    if modelo is None or not columnas:
        raise ValueError(
            "El archivo del modelo no contiene el modelo "
            "y las características esperadas."
        )

    return modelo, columnas


def analizar_paquete(paquete, version):
    """
    Analiza un paquete y devuelve los resultados como diccionario.
    Esta función puede utilizarse desde Flask.
    """
    modelo, columnas = cargar_modelo()

    caracteristicas = obtener_datos_npm(paquete, version)

    faltantes = set(columnas) - set(caracteristicas)

    if faltantes:
        raise ValueError(
            "Faltan características para el modelo: "
            + ", ".join(sorted(faltantes))
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

    auditoria = auditar_vulnerabilidades(paquete, version)

    if auditoria["estado"] != "completado":
        resultado_general = "ANÁLISIS INCOMPLETO"
        mensaje = (
            "No se pudo completar la auditoría de npm. "
            "No se puede concluir que el paquete sea seguro. "
            + auditoria.get("mensaje", "")
        )

    elif auditoria["total"] > 0:
        resultado_general = "RIESGO"
        mensaje = "npm audit reportó vulnerabilidades conocidas."

    elif prediccion == 1:
        resultado_general = "RIESGO SEGÚN EL MODELO"
        mensaje = (
            "El modelo detectó señales de riesgo, aunque npm audit "
            "no reportó vulnerabilidades conocidas."
        )

    else:
        resultado_general = "SIN RIESGO DETECTADO"
        mensaje = (
            "npm audit no reportó vulnerabilidades conocidas "
            "en esta auditoría."
        )

    return {
        "paquete": paquete,
        "version": version,
        "resultado_general": resultado_general,
        "clasificacion_ml": (
            "RIESGO" if prediccion == 1
            else "SIN RIESGO DETECTADO"
        ),
        "probabilidad_riesgo": round(probabilidad * 100, 1),
        "auditoria": auditoria["estado"],
        "paquetes_afectados": (
            len(auditoria.get("avisos", {}))
            if auditoria["estado"] == "completado"
            else 0
        ),
        "criticas": auditoria.get("criticas", 0),
        "altas": auditoria.get("altas", 0),
        "moderadas": auditoria.get("moderadas", 0),
        "bajas": auditoria.get("bajas", 0),
        "mensaje": mensaje,
        "fecha_consulta": datetime.now().strftime(
            "%d/%m/%Y %H:%M:%S"
        ),
        "avisos": auditoria.get("avisos", {}),
        "detalle_auditoria": auditoria.get("mensaje", ""),
    }


def detectar_riesgo():
    """Modo de ejecución por consola."""
    print("\n=== DETECTOR DE RIESGOS OSS ===")

    if not RUTA_MODELO.is_file():
        print(f"No se encontró el modelo: {RUTA_MODELO}")
        return

    paquete = input("Nombre del paquete npm: ").strip()
    version = input("Versión exacta: ").strip()

    if not paquete or not version:
        print("El nombre y la versión son obligatorios.")
        return

    try:
        print("\nAnalizando el paquete...")
        resultado = analizar_paquete(paquete, version)

    except HTTPError as error:
        print(f"Error HTTP consultando npm: {error.code}")
        return
    except URLError as error:
        print(f"No se pudo conectar con npm: {error.reason}")
        return
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        AttributeError,
    ) as error:
        print(f"No se pudo ejecutar el análisis: {error}")
        return

    print("\n=== RESULTADO DEL ANÁLISIS ===")
    print(f"Paquete: {resultado['paquete']}@{resultado['version']}")
    print(f"Resultado general: {resultado['resultado_general']}")
    print(f"Clasificación ML: {resultado['clasificacion_ml']}")
    print(
        "Probabilidad estimada de riesgo: "
        f"{resultado['probabilidad_riesgo']:.1f}%"
    )

    print("\n=== AUDITORÍA npm ===")
    print(f"Estado: {resultado['auditoria']}")

    if resultado["auditoria"] == "completado":
        print(
            "Paquetes afectados: "
            f"{resultado['paquetes_afectados']}"
        )
        print(f"Críticas: {resultado['criticas']}")
        print(f"Altas: {resultado['altas']}")
        print(f"Moderadas: {resultado['moderadas']}")
        print(f"Bajas: {resultado['bajas']}")

        for nombre, aviso in resultado["avisos"].items():
            severidad = aviso.get("severity", "desconocida")
            rango = aviso.get("range", "no especificado")
            print(
                f"- {nombre}: severidad {severidad}, rango {rango}"
            )
    else:
        print(resultado["detalle_auditoria"])

    print(f"\n{resultado['mensaje']}")
    print(
        "\nAviso: npm audit no garantiza la ausencia de "
        "vulnerabilidades. La predicción ML tampoco confirma "
        "ni descarta fallos reales."
    )


if __name__ == "__main__":
    try:
        detectar_riesgo()
    except (KeyboardInterrupt, EOFError):
        print("\nAnálisis cancelado.")
