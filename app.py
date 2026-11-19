
from flask import Flask, render_template, request, jsonify
from pathlib import Path
from datetime import datetime, timezone
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from urllib.parse import quote
import joblib
import json
import os
import re
import shutil
import subprocess
import tempfile

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

BASE_DIR = Path(__file__).resolve().parent
RUTA_MODELO = BASE_DIR / "models" / "random_forest.pkl"

# npm puede estar en el PATH o instalado en la ruta habitual de Windows.
NPM = shutil.which("npm") or shutil.which("npm.cmd")
if not NPM and os.name == "nt":
    ruta_windows = Path(r"C:\Program Files\nodejs\npm.cmd")
    if ruta_windows.exists():
        NPM = str(ruta_windows)

# Cargar el modelo y el orden de características guardado durante el entrenamiento.
if not RUTA_MODELO.exists():
    raise FileNotFoundError(
        f"No se encontró el modelo: {RUTA_MODELO}"
    )

datos_modelo = joblib.load(RUTA_MODELO)

if isinstance(datos_modelo, dict) and "modelo" in datos_modelo:
    modelo = datos_modelo["modelo"]
    caracteristicas_modelo = datos_modelo.get(
        "caracteristicas",
        [
            "edad_version_dias",
            "versiones_anteriores",
            "numero_dependencias",
            "numero_dependencias_opcionales",
            "numero_scripts",
            "tiene_repositorio",
            "tiene_licencia",
        ],
    )
else:
    # Compatibilidad si el archivo contiene directamente el estimador.
    modelo = datos_modelo
    caracteristicas_modelo = [
        "edad_version_dias",
        "versiones_anteriores",
        "numero_dependencias",
        "numero_dependencias_opcionales",
        "numero_scripts",
        "tiene_repositorio",
        "tiene_licencia",
    ]

print("Modelo de OSS Guard cargado correctamente.")


def obtener_json(url, timeout=20):
    """Consulta una URL y devuelve su contenido JSON."""
    req = Request(
        url,
        headers={"User-Agent": "OSS-Guard/1.0"}
    )

    with urlopen(req, timeout=timeout) as respuesta:
        return json.loads(respuesta.read().decode("utf-8"))


def obtener_metadatos(paquete, version):
    """Obtiene los metadatos de un paquete y su versión en npm."""
    nombre_url = quote(paquete, safe="")
    url_paquete = f"https://registry.npmjs.org/{nombre_url}"

    metadatos = obtener_json(url_paquete)

    versiones = metadatos.get("versions", {})
    if version not in versiones:
        raise ValueError(
            f"No se encontró la versión {version} del paquete {paquete}."
        )

    info_version = versiones[version]
    fechas = metadatos.get("time", {})
    fecha_version = fechas.get(version)

    if fecha_version:
        fecha = datetime.fromisoformat(
            fecha_version.replace("Z", "+00:00")
        )
        if fecha.tzinfo is None:
            fecha = fecha.replace(tzinfo=timezone.utc)
        edad_dias = max(
            0,
            (datetime.now(timezone.utc) - fecha).days
        )
    else:
        edad_dias = 0

    # Número de versiones publicadas antes de la versión seleccionada.
    lista_versiones = list(versiones.keys())
    try:
        versiones_anteriores = lista_versiones.index(version)
    except ValueError:
        versiones_anteriores = 0

    dependencias = info_version.get("dependencies") or {}
    opcionales = info_version.get("optionalDependencies") or {}
    scripts = info_version.get("scripts") or {}

    repositorio = info_version.get("repository")
    tiene_repositorio = bool(
        repositorio.get("url")
        if isinstance(repositorio, dict)
        else repositorio
    )

    licencia = info_version.get("license")
    tiene_licencia = bool(licencia)

    caracteristicas = {
        "edad_version_dias": edad_dias,
        "versiones_anteriores": versiones_anteriores,
        "numero_dependencias": len(dependencias),
        "numero_dependencias_opcionales": len(opcionales),
        "numero_scripts": len(scripts),
        "tiene_repositorio": int(tiene_repositorio),
        "tiene_licencia": int(tiene_licencia),
    }

    return caracteristicas


def predecir_riesgo(caracteristicas):
    """Ejecuta la predicción con el modelo entrenado."""
    valores = [
        caracteristicas.get(nombre, 0)
        for nombre in caracteristicas_modelo
    ]

    prediccion = modelo.predict([valores])[0]
    clases = list(getattr(modelo, "classes_", []))

    probabilidad = None
    if hasattr(modelo, "predict_proba"):
        probabilidades = modelo.predict_proba([valores])[0]

        # Identifica la clase de riesgo cuando el modelo la tiene etiquetada.
        indice_riesgo = None
        for i, clase in enumerate(clases):
            etiqueta = str(clase).strip().upper()
            if etiqueta in {"RIESGO", "RISK", "1", "TRUE"}:
                indice_riesgo = i
                break

        if indice_riesgo is not None:
            probabilidad = float(probabilidades[indice_riesgo])
        elif len(probabilidades) > 1:
            # Si las etiquetas no identifican explícitamente la clase,
            # se usa la segunda clase como convención binaria.
            probabilidad = float(probabilidades[1])
        else:
            probabilidad = float(probabilidades[0])

    etiqueta_prediccion = str(prediccion).strip().upper()
    es_riesgo = etiqueta_prediccion in {
        "RIESGO", "RISK", "1", "TRUE"
    }

    # Si las etiquetas del modelo son texto, se respeta la predicción.
    # Para etiquetas numéricas binarias, 1 representa riesgo.
    clasificacion = "RIESGO" if es_riesgo else "SIN RIESGO DETECTADO"

    if probabilidad is None:
        probabilidad = 1.0 if es_riesgo else 0.0

    return clasificacion, max(0.0, min(1.0, probabilidad))


def ejecutar_npm(argumentos, carpeta, timeout=120):
    """Ejecuta npm en Windows o Linux y devuelve su salida."""
    if not NPM:
        raise RuntimeError(
            "No se encontró npm. Instala Node.js o configura npm en el PATH."
        )

    if os.name == "nt":
        comando = subprocess.list2cmdline([NPM, *argumentos])
        return subprocess.run(
            comando,
            cwd=str(carpeta),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            shell=True,
        )

    return subprocess.run(
        [NPM, *argumentos],
        cwd=str(carpeta),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        shell=False,
    )


MAPA_SEVERIDAD = {
    "critical": "Crítica",
    "high": "Alta",
    "moderate": "Moderada",
    "low": "Baja",
    "info": "Informativa",
}


def explicar_impacto_vulnerabilidad(titulo):
    """Genera una explicación sencilla y en español del tipo de vulnerabilidad."""
    t = str(titulo or "").lower()
    if "bracket notation" in t and "memory exhaustion" in t:
        return "Agotamiento de memoria (DoS): Peticiones manipuladas provocan un consumo excesivo de memoria RAM hasta colapsar el servicio."
    if "arraylimit bypass" in t or "limit bypass" in t:
        return "Evasión de límites y saturación (DoS): Permite saltarse las restricciones de tamaño de parámetros para sobrecargar el servidor."
    if "regular expression denial of service" in t or "redos" in t:
        return "Colapso de CPU por expresiones regulares (ReDoS): Rutas maliciosas congelan el procesamiento del servidor."
    if "trailing bytes" in t:
        return "Denegación de servicio (DoS): Peticiones con bytes adicionales tras datos comprimidos causan excepciones no controladas."
    if "invalid limit value" in t:
        return "Fallo de validación de tamaño: Valores inválidos desactivan el límite de peso de peticiones permitiendo sobrecarga."
    if "denial of service" in t or "dos" in t:
        return "Denegación de servicio (DoS): Un atacante puede enviar datos manipulados para colgar o ralentizar el servidor."
    if "remote code execution" in t or "rce" in t or "arbitrary code" in t:
        return "Ejecución de código arbitrario (RCE): Permite a un atacante ejecutar comandos maliciosos dentro de tu servidor."
    if "prototype pollution" in t:
        return "Contaminación de prototipo: Permite inyectar propiedades no autorizadas en objetos de JavaScript afectando la lógica."
    if "path traversal" in t or "directory traversal" in t:
        return "Acceso indebido a archivos (Path Traversal): Permite leer o descargar archivos privados del sistema."
    if "cross-site scripting" in t or "xss" in t:
        return "Inyección de scripts (XSS): Permite inyectar código malicioso en las respuestas del servidor."
    if "open redirect" in t:
        return "Redirección abierta: Permite engañar a usuarios redirigiéndolos a sitios fraudulentos."
    if "information exposure" in t or "information disclosure" in t or "leak" in t:
        return "Fuga de información: Datos confidenciales o internos del servidor podrían quedar expuestos."
    return "Fallo de seguridad reportado que puede comprometer la estabilidad o protección del sistema."


def explicar_solucion_rango(rango):
    """Explica en español qué versiones se ven afectadas y en cuál se soluciona."""
    if not rango or rango == "No especificado":
        return "Versión de solución no especificada."
    r = str(rango).strip()
    if r.startswith("<="):
        ver = r[2:].strip()
        return f"Afecta hasta la versión {ver}. Se resuelve actualizando a una versión posterior."
    if r.startswith("<"):
        ver = r[1:].strip()
        return f"Afecta a versiones anteriores a {ver}. Se resuelve en {ver} o superior."
    if "||" in r:
        return f"Afecta a múltiples rangos ({r}). Se recomienda actualizar a la última versión disponible."
    return f"Afecta al rango {r}. Actualiza fuera de este rango."


def auditar_dependencia(paquete, version):
    """Ejecuta npm audit para una dependencia concreta."""
    resultado_vacio = {
        "auditoria": "NO DISPONIBLE",
        "paquetes_afectados": 0,
        "criticas": 0,
        "altas": 0,
        "moderadas": 0,
        "bajas": 0,
        "detalles_vulnerabilidades": [],
    }

    if not NPM:
        return resultado_vacio

    try:
        with tempfile.TemporaryDirectory(prefix="oss_guard_") as temporal:
            carpeta = Path(temporal)

            package_json = {
                "name": "oss-guard-audit",
                "version": "1.0.0",
                "private": True,
                "dependencies": {
                    paquete: version
                },
            }

            (carpeta / "package.json").write_text(
                json.dumps(package_json, indent=2),
                encoding="utf-8",
            )

            # Genera el lockfile sin ejecutar scripts de instalación.
            instalacion = ejecutar_npm(
                [
                    "install",
                    "--package-lock-only",
                    "--ignore-scripts",
                    "--no-audit",
                    "--no-fund",
                ],
                carpeta,
            )

            if not (carpeta / "package-lock.json").exists():
                detalle = (
                    instalacion.stderr or instalacion.stdout or ""
                ).strip()
                if detalle:
                    print("No se pudo generar package-lock.json:", detalle)
                return resultado_vacio

            auditoria = ejecutar_npm(
                ["audit", "--json"],
                carpeta,
                timeout=120,
            )

            # npm audit suele devolver código 1 cuando encuentra avisos.
            try:
                datos = json.loads(auditoria.stdout or "{}")
            except json.JSONDecodeError:
                detalle = (
                    auditoria.stderr or auditoria.stdout or ""
                ).strip()
                if detalle:
                    print("Respuesta de npm audit no válida:", detalle)
                return resultado_vacio

            resumen = datos.get("metadata", {}).get(
                "vulnerabilities", {}
            )

            criticas = int(resumen.get("critical", 0) or 0)
            altas = int(resumen.get("high", 0) or 0)
            moderadas = int(resumen.get("moderate", 0) or 0)
            bajas = int(resumen.get("low", 0) or 0)

            vulnerabilidades = datos.get("vulnerabilities") or {}
            detalles = []

            for nombre, info in vulnerabilidades.items():
                via = info.get("via") or []

                # "via" puede contener texto o detalles estructurados.
                entradas = [
                    elemento for elemento in via
                    if isinstance(elemento, dict)
                ]

                if not entradas:
                    entradas = [{}]

                for entrada in entradas:
                    titulo = entrada.get("title") or (
                        f"Vulnerabilidad relacionada con {nombre}"
                    )
                    severidad_raw = str(
                        entrada.get("severity")
                        or info.get("severity")
                        or "desconocida"
                    ).lower()
                    rango = str(
                        entrada.get("range")
                        or info.get("range")
                        or "No especificado"
                    )

                    detalles.append({
                        "titulo": titulo,
                        "impacto": explicar_impacto_vulnerabilidad(titulo),
                        "paquete": nombre,
                        "rol_paquete": (
                            "Paquete principal consultado"
                            if nombre == paquete
                            else f"Dependencia interna de {paquete}"
                        ),
                        "es_principal": bool(nombre == paquete),
                        "severidad_raw": severidad_raw,
                        "severidad": MAPA_SEVERIDAD.get(severidad_raw, severidad_raw.capitalize()),
                        "rango": rango,
                        "solucion": explicar_solucion_rango(rango),
                    })

            total = int(
                resumen.get("total", 0)
                or (criticas + altas + moderadas + bajas)
            )

            return {
                "auditoria": (
                    "VULNERABILIDADES DETECTADAS"
                    if total > 0
                    else "SIN VULNERABILIDADES CONOCIDAS"
                ),
                "paquetes_afectados": total,
                "criticas": criticas,
                "altas": altas,
                "moderadas": moderadas,
                "bajas": bajas,
                "detalles_vulnerabilidades": detalles,
            }

    except subprocess.TimeoutExpired:
        print("npm audit excedió el tiempo máximo.")
        return resultado_vacio
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"No se pudo completar npm audit: {exc}")
        return resultado_vacio


def analizar_dependencia(paquete, version):
    """Combina la predicción ML y la auditoría npm."""
    caracteristicas = obtener_metadatos(paquete, version)
    clasificacion_ml, probabilidad = predecir_riesgo(caracteristicas)
    auditoria = auditar_dependencia(paquete, version)

    auditoria_disponible = auditoria["auditoria"] != "NO DISPONIBLE"
    hay_vulnerabilidades = auditoria["paquetes_afectados"] > 0

    resultado_general = (
        "RIESGO"
        if clasificacion_ml == "RIESGO" or hay_vulnerabilidades
        else "SIN RIESGO DETECTADO"
    )

    if hay_vulnerabilidades and clasificacion_ml == "RIESGO":
        mensaje = (
            "Riesgo alto: el paquete presenta anomalías estructurales y fallos de seguridad confirmados. "
            "Se desaconseja totalmente su instalación."
        )
    elif hay_vulnerabilidades:
        mensaje = (
            "Esta versión contiene fallos de seguridad conocidos en sus dependencias. "
            "No se recomienda su uso en producción; actualiza a una versión más reciente donde estén corregidos."
        )
    elif clasificacion_ml == "RIESGO":
        mensaje = (
            "Alerta preventiva: el paquete presenta características inusuales en sus scripts o estructura. "
            "Se recomienda revisarlo con precaución antes de instalarlo."
        )
    elif not auditoria_disponible:
        mensaje = (
            "El modelo predictivo no detectó anomalías, pero la auditoría oficial "
            "no estuvo disponible. Revisa manualmente la versión antes de usarla."
        )
    else:
        mensaje = (
            "El paquete superó la auditoría oficial y la evaluación predictiva. "
            "No se detectaron fallos de seguridad ni señales de riesgo."
        )

    return {
        "paquete": paquete,
        "version": version,
        "resultado_general": resultado_general,
        "clasificacion_ml": clasificacion_ml,
        "probabilidad_riesgo": probabilidad,
        **auditoria,
        "mensaje": mensaje,
        "fecha_consulta": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        "caracteristicas": caracteristicas,
    }


@app.route("/", methods=["GET"])
def inicio():
    return render_template(
        "index.html",
        resultado=None,
        error=None,
        paquete="",
        version="",
    )


@app.route("/analizar", methods=["POST"])
def analizar():
    paquete = request.form.get("paquete", "").strip()
    version = request.form.get("version", "").strip()

    if not paquete or not version:
        return render_template(
            "index.html",
            resultado=None,
            error="Introduce el nombre del paquete y su versión exacta.",
            paquete=paquete,
            version=version,
        )

    # Validación básica: evita espacios y caracteres extraños en la entrada.
    if not re.fullmatch(r"@?[a-zA-Z0-9._~-]+(?:/[a-zA-Z0-9._~-]+)?", paquete):
        return render_template(
            "index.html",
            resultado=None,
            error="El nombre del paquete npm no tiene un formato válido.",
            paquete=paquete,
            version=version,
        )

    if not re.fullmatch(r"[a-zA-Z0-9.+_-]+", version):
        return render_template(
            "index.html",
            resultado=None,
            error="Introduce una versión concreta, por ejemplo 4.21.2.",
            paquete=paquete,
            version=version,
        )

    try:
        resultado = analizar_dependencia(paquete, version)
        return render_template(
            "index.html",
            resultado=resultado,
            error=None,
            paquete=paquete,
            version=version,
        )
    except HTTPError as exc:
        if exc.code == 404:
            mensaje = (
                f"No se encontró el paquete '{paquete}' o la versión "
                f"'{version}' en el registro de npm."
            )
        else:
            mensaje = f"El registro de npm devolvió el error HTTP {exc.code}."
    except URLError:
        mensaje = (
            "No se pudo conectar con el registro de npm. "
            "Comprueba tu conexión a Internet."
        )
    except (ValueError, KeyError, TypeError) as exc:
        mensaje = str(exc) or "Los datos de la dependencia no son válidos."
    except Exception as exc:
        app.logger.exception("Error durante el análisis")
        mensaje = f"No se pudo completar el análisis: {exc}"

    return render_template(
        "index.html",
        resultado=None,
        error=mensaje,
        paquete=paquete,
        version=version,
    )


@app.route("/api/analizar", methods=["POST"])
def api_analizar():
    """Endpoint JSON: recibe {"paquete": "express", "version": "4.21.2"}."""
    datos = request.get_json(silent=True) or {}
    paquete = str(datos.get("paquete", "")).strip()
    version = str(datos.get("version", "")).strip()

    if not paquete or not version:
        return jsonify({
            "error": "Debes enviar los campos paquete y version."
        }), 400

    try:
        resultado = analizar_dependencia(paquete, version)
        return jsonify(resultado)
    except HTTPError as exc:
        return jsonify({
            "error": f"El registro de npm devolvió HTTP {exc.code}."
        }), exc.code
    except URLError:
        return jsonify({
            "error": "No se pudo conectar con el registro de npm."
        }), 502
    except Exception:
        app.logger.exception("Error en la API de análisis")
        return jsonify({
            "error": "No se pudo completar el análisis."
        }), 500


if __name__ == "__main__":
    app.run(debug=True)
