import requests

paquete = "lodash"
version = "4.17.20"

url = "https://api.osv.dev/v1/query"

consulta = {
    "package": {
        "name": paquete,
        "ecosystem": "npm"
    },
    "version": version
}

respuesta = requests.post(
    url,
    json=consulta,
    timeout=20
)

if respuesta.status_code == 200:

    datos = respuesta.json()
    vulnerabilidades = datos.get("vulns", [])

    print("===================================")
    print(" PRUEBA DE VERSION VULNERABLE")
    print("===================================")

    print("Paquete:", paquete)
    print("Versión:", version)
    print(
        "Vulnerabilidades:",
        len(vulnerabilidades)
    )

    for vulnerabilidad in vulnerabilidades:
        print(
            "-",
            vulnerabilidad.get("id"),
            ":",
            vulnerabilidad.get("summary")
        )

else:

    print("Error al consultar OSV")
    print("Código:", respuesta.status_code)