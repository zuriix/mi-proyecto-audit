
from pathlib import Path
from datetime import datetime, timezone
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from urllib.parse import quote
import json

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
RUTA_MODELO = BASE_DIR / "models" / "random_forest.pkl"
URL_NPM = "https://registry.npmjs.org"


def obtener_datos_npm(paquete, version):
    """Obtiene metadatos reales de una version publicada en npm."""
    nombre_url = quote(paquete, safe="@")
    url = f"{URL_NPM}/{nombre_url}"

    solicitud = Request(
        url,
        headers={"User-Agent": "DetectorRiesgosOSS/1.0"}
    )

    with urlopen(solicitud, timeout=20) as respuesta:
        metadatos = json.load(respuesta)

    versiones = metadatos.get("versions", {})
    if version not in versiones:
        raise ValueError(
            f"No se encontro la version {version} del paquete {paquete}."
        )

    datos_version = versiones[version]
    fechas = metadatos.get("time", {})
    fecha_publicacion = fechas.get(version)

    if not fecha_publicacion:
        raise ValueError(
            "npm no proporciona la fecha de publicacion de esta version."
        )

    fecha = datetime.fromisoformat(
        fecha_publicacion.replace("Z", "+00:00")
    )
    ahora = datetime.now(timezone.utc)

    edad = max(0, (ahora - fecha).total_seconds() / 86400)

    versiones_anteriores = 0
    for nombre, fecha_texto in fechas.items():
        if nombre in ("created", "modified"):
            continue
        if nombre == version or not isinstance(fecha_texto, str):
            continue

        try:
            fecha_otra = datetime.fromisoformat(
                fecha_texto.replace("Z", "+00:00")
            )
            if fecha_otra < fecha:
                versiones_anteriores += 1
        except ValueError:
            continue

    dependencias = datos_version.get("dependencies") or {}
    opcionales = datos_version.get("optionalDependencies") or {}
    scripts = datos_version.get("scripts") or {}
    repositorio = datos_version.get("repository")
    licencia = datos_version.get("license")

    caracteristicas = {
        "edad_version_dias": edad,
        "versiones_anteriores": versiones_anteriores,
        "numero_dependencias": len(dependencias),
        "numero_dependencias_opcionales": len(opcionales),
        "numero_scripts": len(scripts),
        "tiene_repositorio": int(bool(repositorio)),
        "tiene_licencia": int(bool(licencia)),
    }

    return caracteristicas


def detectar_riesgo():
    print("\n=== DETECTOR DE RIESGOS OSS ===")

    if not RUTA_MODELO.is_file():
        print(f"No se encontro el modelo: {RUTA_MODELO}")
        print("Comprueba que exista models/random_forest.pkl.")
        return

    paquete = input("Nombre del paquete npm: ").strip()
    version = input("Version exacta del paquete: ").strip()

    if not paquete or not version:
        print("Debes introducir el nombre y la version.")
        return

    try:
        print("\nConsultando los datos de npm...")
        caracteristicas_datos = obtener_datos_npm(paquete, version)

        artefacto = joblib.load(RUTA_MODELO)
        modelo = artefacto["modelo"]
        caracteristicas_modelo = artefacto["caracteristicas"]

        faltantes = (
            set(caracteristicas_modelo) - set(caracteristicas_datos)
        )
        if faltantes:
            print(
                "El modelo requiere caracteristicas no disponibles: "
                f"{sorted(faltantes)}"
            )
            return

        X = pd.DataFrame([{
            nombre: caracteristicas_datos[nombre]
            for nombre in caracteristicas_modelo
        }])

        prediccion = int(modelo.predict(X)[0])
        clases = list(modelo.classes_)
        probabilidades = modelo.predict_proba(X)[0]

        if 1 in clases:
            probabilidad = float(probabilidades[clases.index(1)])
        else:
            probabilidad = 0.0

    except HTTPError as error:
        if error.code == 404:
            print("No se encontro el paquete en npm.")
        else:
            print(f"Error HTTP al consultar npm: {error.code}")
        return
    except URLError as error:
        print(f"No se pudo conectar con npm: {error.reason}")
        return
    except (OSError, ValueError, KeyError, TypeError,
            AttributeError, json.JSONDecodeError) as error:
        print(f"No se pudo completar el analisis: {error}")
        return

    print("\n" + "=" * 44)
    print("RESULTADO DEL ANALISIS")
    print("=" * 44)
    print(f"Paquete: {paquete}")
    print(f"Version: {version}")
    print(
        "Clasificacion: "
        + ("RIESGO" if prediccion == 1 else "SIN RIESGO DETECTADO")
    )
    print(f"Probabilidad estimada de riesgo: {probabilidad:.1%}")

    print("\nCaracteristicas obtenidas de npm:")
    for nombre, valor in caracteristicas_datos.items():
        if nombre == "edad_version_dias":
            print(f"  {nombre}: {valor:.1f}")
        else:
            print(f"  {nombre}: {valor}")

    print("\nIMPORTANTE:")
    print("El resultado es una prediccion del modelo.")
    print("No confirma ni descarta vulnerabilidades reales.")
    print("Los datos se obtienen de los metadatos de npm.")


if __name__ == "__main__":
    try:
        detectar_riesgo()
    except (KeyboardInterrupt, EOFError):
        print("\nAnalisis cancelado.")
