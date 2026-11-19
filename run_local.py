"""
Script de inicio local para OSS Guard.
Configura las rutas de dependencias necesarias y arranca el servidor Flask en http://127.0.0.1:5000.
"""

from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent
VENV_SITE_PACKAGES = BASE_DIR / "venv" / "Lib" / "site-packages"

if VENV_SITE_PACKAGES.exists() and str(VENV_SITE_PACKAGES) not in sys.path:
    # Agregar al final para priorizar las librerías binarias nativas instaladas en el sistema
    sys.path.append(str(VENV_SITE_PACKAGES))

import app

if __name__ == "__main__":
    print()
    print("=" * 60)
    print(" INICIANDO SERVIDOR LOCAL DE OSS GUARD")
    print(" URL: http://127.0.0.1:5000")
    print(" Presiona Ctrl + C para detener el servidor.")
    print("=" * 60)
    print()
    app.app.run(host="127.0.0.1", port=5000, debug=False)
