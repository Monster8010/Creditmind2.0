# CreditMind

Prototipo académico para evaluar riesgo de incumplimiento crediticio mediante:

- Regresión Logística.
- Random Forest.
- XGBoost.
- Red Neuronal.

La aplicación utiliza modelos previamente entrenados. Al ejecutarla, los modelos
se cargan desde archivos persistidos y **no vuelven a entrenarse**.

## Funciones

- Evaluación de solicitantes.
- Selección de un modelo o comparación de los cuatro.
- Probabilidad estimada de default.
- Decisión orientativa y nivel de riesgo.
- Métricas por modelo.
- Matrices de confusión.
- Importancia e interpretación de variables.

## Archivos en GitHub y Google Drive

El código fuente se almacena en GitHub.

Los archivos pesados se distribuyen mediante Google Drive:

- `Loan_default_limpio.csv`
- `logistic_regression.pkl`
- `random_forest.pkl`
- `xgboost.pkl`
- `neural_network.pkl`

Enlace de Google Drive:

```text
https://drive.google.com/file/d/1XHzz3tCLkC05c8U4HN5FXOY1jzsqIRdT/view?usp=drive_link
```

## Requisitos

- Python 3.12.
- PowerShell en Windows.
- Aproximadamente 500 MB libres para dependencias y artefactos.

Versiones principales:

```text
scikit-learn 1.8.0
numpy        2.4.3
xgboost      3.2.0
```

## Instalación desde GitHub

### 1. Clonar el repositorio

```powershell
git clone https://github.com/Monster8010/Creditmind.git
cd Creditmind
```

También se puede descargar el repositorio desde GitHub usando:

```text
Code > Download ZIP
```

Después se extrae el ZIP y se abre PowerShell dentro de la carpeta extraída.

### 2. Descargar los archivos pesados

Descarga `CreditMind_Drive_Package.zip` desde el enlace de Google Drive indicado
anteriormente.

Extrae el ZIP. Su contenido debe ser:

```text
Loan_default_limpio.csv
artifacts/
├── features.json
├── model_registry_notebook.json
└── trained_models/
    ├── logistic_regression.pkl
    ├── random_forest.pkl
    ├── xgboost.pkl
    └── neural_network.pkl
```

Copia `Loan_default_limpio.csv` y la carpeta `artifacts` dentro de la carpeta
del repositorio. Acepta combinar o reemplazar archivos cuando Windows lo
solicite.

La estructura final debe ser:

```text
Creditmind/
├── app/
│   └── main.py
├── artifacts/
│   ├── features.json
│   ├── model_registry_notebook.json
│   └── trained_models/
│       ├── logistic_regression.pkl
│       ├── random_forest.pkl
│       ├── xgboost.pkl
│       └── neural_network.pkl
├── Loan_default_limpio.csv
├── app.js
├── index.html
├── styles.css
├── requirements.txt
└── README.md
```

### 3. Crear el entorno virtual

```powershell
python -m venv .venv
```

No es obligatorio activar el entorno. Los comandos siguientes llaman
directamente a su Python.

### 4. Instalar dependencias

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install --timeout 1000 --retries 10 -r requirements.txt
```

Verifica las versiones:

```powershell
.\.venv\Scripts\python.exe -c "import sklearn, numpy, xgboost; print('sklearn:', sklearn.__version__); print('numpy:', numpy.__version__); print('xgboost:', xgboost.__version__)"
```

## Ejecución

Se necesitan dos terminales abiertas en la carpeta del repositorio.

### Terminal 1: backend

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Comprueba el backend:

```text
http://127.0.0.1:8000/health
```

Documentación de la API:

```text
http://127.0.0.1:8000/docs
```

### Terminal 2: frontend

```powershell
.\.venv\Scripts\python.exe -m http.server 8080 --bind 127.0.0.1
```

Abre la aplicación:

```text
http://127.0.0.1:8080
```

Si el navegador muestra una versión anterior, utiliza `Ctrl + F5`.

## Detener la aplicación

Presiona `Ctrl + C` en cada terminal.

## Crear el paquete para Google Drive

El responsable del proyecto puede generar automáticamente el ZIP de archivos
pesados ejecutando:

```powershell
powershell -ExecutionPolicy Bypass -File .\prepare_drive_package.ps1
```

Se creará:

```text
CreditMind_Drive_Package.zip
```

El ZIP contiene el dataset, el registro de modelos, la lista de variables y los
cuatro modelos entrenados. Después sólo debe subirse a Google Drive.

## Publicar el código en GitHub

Los modelos y datasets están excluidos mediante `.gitignore`, por lo que no se
subirán accidentalmente.

```powershell
git add .
git status
git commit -m "Actualiza CreditMind y documentacion"
git push -u origin main
```

Antes del commit verifica que no aparezcan archivos `.pkl`, `.csv`, `.env` ni
la carpeta `.venv`.

## Configurar Google Drive

1. Ejecuta `prepare_drive_package.ps1`.
2. Abre Google Drive.
3. Sube `CreditMind_Drive_Package.zip`.
4. Haz clic derecho sobre el archivo y selecciona `Compartir`.
5. Cambia el acceso a `Cualquier persona con el enlace`.
6. Selecciona permiso de `Lector`.
7. Copia el enlace.
8. Sustituye en este README:

```text
REEMPLAZAR_CON_EL_ENLACE_PUBLICO_DE_GOOGLE_DRIVE
```

por el enlace real.

Después vuelve a realizar un commit y push:

```powershell
git add README.md
git commit -m "Agrega enlace de descarga de artefactos"
git push
```

## Actualizar modelos

Cuando se reentrenen los modelos:

1. Ejecuta el notebook completo.
2. Exporta los nuevos artefactos.
3. Reemplaza los archivos dentro de `artifacts/trained_models/`.
4. Actualiza `features.json` y `model_registry_notebook.json`.
5. Ejecuta nuevamente `prepare_drive_package.ps1`.
6. Reemplaza el ZIP en Google Drive.
7. Reinicia FastAPI.

## Dataset en otra ubicación

El backend busca `Loan_default_limpio.csv` en:

1. La raíz del repositorio.
2. La carpeta `data/`.
3. La carpeta `artifacts/`.
4. La ruta definida mediante `CREDITMIND_DATASET`.

Ejemplo:

```powershell
$env:CREDITMIND_DATASET="D:\datasets\Loan_default_limpio.csv"
```

## Errores comunes

### API no conectada

Confirma que el backend esté ejecutándose y visita:

```text
http://127.0.0.1:8000/health
```

### Puerto ocupado

```powershell
netstat -ano | findstr :8000
Stop-Process -Id NUMERO_PID -Force
```

### Error al cargar modelos

Comprueba que:

- Los cuatro `.pkl` estén en `artifacts/trained_models/`.
- `model_registry_notebook.json` exista.
- Las versiones de `scikit-learn`, `numpy` y `xgboost` sean compatibles.

## Nota académica

CreditMind es un prototipo académico. Sus resultados no deben utilizarse como
único criterio para aprobar o rechazar créditos reales.
