import requests

paquete = "lodash"

url = "https://api.osv.dev/v1/query"

consulta = {
    "package": {
        "name": paquete,
        "ecosystem": "npm"
    }
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
    print(" VULNERABILIDADES DEL PAQUETE")
    print("===================================")

    print("Paquete:", paquete)
    print("Vulnerabilidades encontradas:", len(vulnerabilidades))

    for vulnerabilidad in vulnerabilidades:

        print("\n-----------------------------------")

        print("ID:", vulnerabilidad.get("id"))
        print("Resumen:", vulnerabilidad.get("summary"))

        afectados = vulnerabilidad.get("affected", [])

        for afectado in afectados:

            rangos = afectado.get("ranges", [])

            for rango in rangos:

                eventos = rango.get("events", [])

                print("Eventos de versión:")

                for evento in eventos:

                    print(evento)

else:

    print("Error al consultar OSV")
    print("Código:", respuesta.status_code)
    print(respuesta.text)