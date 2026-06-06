"""Descarga y extrae los artefactos pesados durante el despliegue."""

import os
import zipfile
from pathlib import Path

import gdown


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_FILE_ID = "1XHzz3tCLkC05c8U4HN5FXOY1jzsqIRdT"
FILE_ID = os.getenv("CREDITMIND_DRIVE_FILE_ID", DEFAULT_FILE_ID)
ZIP_PATH = BASE_DIR / ".creditmind_artifacts.zip"
REQUIRED_MODEL = BASE_DIR / "artifacts" / "trained_models" / "xgboost.pkl"


def main() -> None:
    if REQUIRED_MODEL.exists():
        print("Los artefactos ya existen; se omite la descarga.")
        return

    print("Descargando artefactos de CreditMind desde Google Drive...")
    result = gdown.download(
        id=FILE_ID,
        output=str(ZIP_PATH),
        quiet=False,
        fuzzy=True,
    )

    if not result or not ZIP_PATH.exists():
        raise RuntimeError("Google Drive no devolvió el paquete de artefactos.")

    if not zipfile.is_zipfile(ZIP_PATH):
        raise RuntimeError("El archivo descargado no es un ZIP válido.")

    with zipfile.ZipFile(ZIP_PATH) as package:
        package.extractall(BASE_DIR)

    ZIP_PATH.unlink(missing_ok=True)

    if not REQUIRED_MODEL.exists():
        raise RuntimeError(
            "El ZIP no contiene artifacts/trained_models/xgboost.pkl."
        )

    print("Artefactos descargados y extraídos correctamente.")


if __name__ == "__main__":
    main()
