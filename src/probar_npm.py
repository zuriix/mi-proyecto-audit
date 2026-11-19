
import subprocess
from pathlib import Path

NPM = Path(r"C:\Program Files\nodejs\npm.cmd")


def comprobar_npm():
    if not NPM.is_file():
        print("No se encontro npm.cmd.")
        return

    try:
        resultado = subprocess.run(
            [str(NPM), "--version"],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
        print("npm funciona desde Python.")
        print("Version:", resultado.stdout.strip())

    except (subprocess.SubprocessError, OSError) as error:
        print("No se pudo ejecutar npm:", error)


if __name__ == "__main__":
    comprobar_npm()
