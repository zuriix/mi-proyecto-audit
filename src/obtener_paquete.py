import requests
import pandas as pd
import os
import time


# ==========================================
# CONFIGURACIÓN
# ==========================================

PAQUETES = [
    "lodash",
    "express",
    "axios",
    "react",
    "moment",
    "chalk",
    "commander",
    "uuid",
    "jsonwebtoken",
    "mongoose",
    "jquery",
    "webpack",
    "typescript",
    "eslint",
    "prettier",
    "next",
    "vue",
    "rxjs",
    "socket.io",
    "request",
    "underscore",
    "debug",
    "semver",
    "yargs",
    "glob",
    "minimist",
    "qs",
    "body-parser",
    "cors",
    "dotenv",
    "joi",
    "validator",
    "bcrypt",
    "passport",
    "passport-local",
    "cookie-parser",
    "multer",
    "node-fetch",
    "inquirer",
    "ora",
    "rimraf",
    "cross-env",
    "husky",
    "nodemon",
    "pm2",
    "http-proxy",
    "tar"
]


URL_NPM = "https://registry.npmjs.org"


ARCHIVO_SALIDA = (
    "data/raw/dataset_npm.csv"
)


# ==========================================
# FUNCIÓN PARA CONSULTAR UN PAQUETE
# ==========================================

def obtener_informacion_paquete(paquete):

    url = f"{URL_NPM}/{paquete}"

    try:

        respuesta = requests.get(
            url,
            timeout=30
        )

        if respuesta.status_code != 200:

            print(
                f"ERROR {respuesta.status_code}: "
                f"{paquete}"
            )

            return None

        datos = respuesta.json()


        # ----------------------------------
        # Versión actual
        # ----------------------------------

        dist_tags = datos.get(
            "dist-tags",
            {}
        )

        version_actual = dist_tags.get(
            "latest"
        )


        if not version_actual:

            print(
                f"No se encontró versión latest: "
                f"{paquete}"
            )

            return None


        # ----------------------------------
        # Información de la versión actual
        # ----------------------------------

        versiones = datos.get(
            "versions",
            {}
        )

        informacion_version = versiones.get(
            version_actual,
            {}
        )


        # ----------------------------------
        # Dependencias
        # ----------------------------------

        dependencias = informacion_version.get(
            "dependencies",
            {}
        )

        dependencias_opcionales = (
            informacion_version.get(
                "optionalDependencies",
                {}
            )
        )


        # ----------------------------------
        # Scripts
        # ----------------------------------

        scripts = informacion_version.get(
            "scripts",
            {}
        )


        # ----------------------------------
        # Repositorio
        # ----------------------------------

        repositorio = datos.get(
            "repository",
            ""
        )


        if isinstance(
            repositorio,
            dict
        ):

            repositorio = repositorio.get(
                "url",
                ""
            )


        # ----------------------------------
        # Licencia
        # ----------------------------------

        licencia = informacion_version.get(
            "license",
            ""
        )


        if isinstance(
            licencia,
            dict
        ):

            licencia = licencia.get(
                "type",
                ""
            )


        # ----------------------------------
        # Información final
        # ----------------------------------

        resultado = {

            "paquete": datos.get(
                "name",
                paquete
            ),

            "version_actual": version_actual,

            "fecha_creacion": datos.get(
                "time",
                {}
            ).get(
                "created"
            ),

            "ultima_actualizacion": datos.get(
                "time",
                {}
            ).get(
                "modified"
            ),

            "numero_versiones": len(
                versiones
            ),

            "numero_dependencias": len(
                dependencias
            ),

            "numero_dependencias_opcionales": len(
                dependencias_opcionales
            ),

            "numero_scripts": len(
                scripts
            ),

            "repositorio": repositorio,

            "licencia": licencia
        }


        return resultado


    except requests.RequestException as error:

        print(
            f"ERROR de conexión con "
            f"{paquete}: {error}"
        )

        return None


    except Exception as error:

        print(
            f"ERROR procesando "
            f"{paquete}: {error}"
        )

        return None


# ==========================================
# INICIO
# ==========================================

print()
print("===================================")
print("RECOLECCIÓN DE PAQUETES NPM")
print("===================================")

print(
    f"Paquetes a consultar: "
    f"{len(PAQUETES)}"
)


resultados = []


# ==========================================
# CONSULTAR PAQUETES
# ==========================================

for numero, paquete in enumerate(
    PAQUETES,
    start=1
):

    print()
    print(
        f"[{numero}/{len(PAQUETES)}] "
        f"Consultando: {paquete}"
    )


    resultado = (
        obtener_informacion_paquete(
            paquete
        )
    )


    if resultado is not None:

        resultados.append(
            resultado
        )

        print(
            f"OK: {paquete} -> "
            f"{resultado['version_actual']}"
        )

        print(
            f"   Versiones disponibles: "
            f"{resultado['numero_versiones']}"
        )

    else:

        print(
            f"NO SE PUDO OBTENER: "
            f"{paquete}"
        )


    # Pausa entre solicitudes
    time.sleep(0.5)


# ==========================================
# CREAR CARPETA
# ==========================================

os.makedirs(
    "data/raw",
    exist_ok=True
)


# ==========================================
# CREAR DATAFRAME
# ==========================================

columnas = [

    "paquete",

    "version_actual",

    "fecha_creacion",

    "ultima_actualizacion",

    "numero_versiones",

    "numero_dependencias",

    "numero_dependencias_opcionales",

    "numero_scripts",

    "repositorio",

    "licencia"

]


df = pd.DataFrame(
    resultados,
    columns=columnas
)


# ==========================================
# ELIMINAR DUPLICADOS
# ==========================================

if not df.empty:

    df = (
        df
        .drop_duplicates(
            subset=["paquete"]
        )
        .reset_index(
            drop=True
        )
    )


# ==========================================
# GUARDAR
# ==========================================

df.to_csv(
    ARCHIVO_SALIDA,
    index=False,
    encoding="utf-8"
)


# ==========================================
# RESUMEN
# ==========================================

print()
print("===================================")
print("RECOLECCIÓN TERMINADA")
print("===================================")

print(
    f"Paquetes solicitados: "
    f"{len(PAQUETES)}"
)

print(
    f"Paquetes obtenidos: "
    f"{len(df)}"
)

print()
print(
    f"Archivo generado:"
)

print(
    ARCHIVO_SALIDA
)

print()
print("===================================")
print("FIN DEL PROCESO")
print("===================================")