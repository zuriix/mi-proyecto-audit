import requests

paquete = "lodash"
version = "4.18.1"

url = "https://api.osv.dev/v1/query"

datos_consulta = {
    "package": {
        "name": paquete,
        "ecosystem": "npm"
    },
    "version": version
}

respuesta = requests.post(
    url,
    json=datos_consulta,
    timeout=15
)

if respuesta.status_code == 200:

    datos = respuesta.json()

    vulnerabilidades = datos.get("vulns", [])

    print("===================================")
    print(" ANÁLISIS DE VULNERABILIDADES")
    print("===================================")

    print("Paquete:", paquete)
    print("Versión:", version)
    print("Vulnerabilidades encontradas:", len(vulnerabilidades))

    if vulnerabilidades:

        print("\nIDs de vulnerabilidades:")

        for vulnerabilidad in vulnerabilidades:
            print("-", vulnerabilidad.get("id"))

    else:

        print("No se encontraron vulnerabilidades conocidas.")

else:

    print("Error al consultar OSV")
    print("Código:", respuesta.status_code)
    print("Respuesta:", respuesta.text)
    