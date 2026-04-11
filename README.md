# Sistema Crediticio — FastAPI Backend
## Guía completa de implementación · Demo académica local

---

## A) ARQUITECTURA RECOMENDADA

```
[Notebook Colab/local]
        │
        │  joblib.dump(best_xgb)  ──→  artifacts/model_xgb.pkl
        │  joblib.dump(best_rf)   ──→  artifacts/model_rf.pkl
        │  json.dump(FEATURES)    ──→  artifacts/features.json
        │
        ▼
[FastAPI · uvicorn · puerto 8000]
   POST /predict  ← recibe JSON con 6 features
   GET  /health   ← verifica que la API está viva
   GET  /model-info
        │
        │  responde JSON en < 50ms
        ▼
[Frontend · index.html + app.js]
   Tab 1: sliders → debounce 250ms → POST /predict → muestra riesgo
   Tab 2: formulario → POST /predict + análisis IA (Anthropic API)
   Tab 3: historial de sesión en memoria
```

---

## B) ¿POR QUÉ NO ENTRENAR EN TIEMPO REAL?

1. **Tiempo**: GridSearchCV sobre 255k registros tarda minutos. El slider debe
   responder en < 300ms.
2. **Recursos**: reentrenar bloquea la CPU de tu máquina y rompe la demo.
3. **Consistencia**: entrenar en vivo produce modelos distintos cada vez.
4. **XGBoost no necesita scaler**: es tree-based → el artefacto pesa ~2MB
   y la inferencia tarda < 2ms.

**Regla de oro**: entrenas una vez, guardas el artefacto, la API solo carga
y sirve. Es el flujo estándar de MLOps.

---

## C) ¿ES NECESARIO DOCKER?

**No para una demo local.** Docker añade 20-30 min de setup, requiere
instalar Docker Desktop y complica el debug. Úsalo solo si despliegas
a un servidor externo.

Para esta demo: Python 3.10+ + pip + uvicorn es suficiente.

---

## D) ESTRUCTURA DE CARPETAS

```
proyecto_crediticio/
│
├── artifacts/                  ← artefactos del modelo (generar con notebook)
│   ├── model_xgb.pkl           ← XGBoost optimizado (AUC 0.720)
│   ├── model_rf.pkl            ← Random Forest (opcional, para comparar)
│   └── features.json           ← ['Age','Income','LoanAmount','CreditScore',
│                                   'MonthsEmployed','NumCreditLines']
│
├── app/
│   └── main.py                 ← FastAPI · endpoints /predict /health /model-info
│
├── index.html                  ← frontend (ya existente, no modificar)
├── styles.css                  ← estilos (ya existente)
├── app.js                      ← lógica frontend actualizada (reemplazar)
│
├── export_artifacts.py         ← copiar+pegar al final del notebook
├── requirements.txt            ← dependencias Python
└── README.md                   ← este archivo
```

---

## H) INSTRUCCIONES PASO A PASO

### Paso 1 — Exportar artefactos desde el notebook

Abre tu notebook `Loan_RandomForest_modificado.ipynb` y **pega al final**
el contenido de `export_artifacts.py` como una nueva celda. Ejecútala.

Verás:
```
✓ model_xgb.pkl  guardado
✓ model_rf.pkl   guardado
✓ features.json  guardado → ['Age', 'Income', ...]
✓ Verificación de carga: prob_default = 0.3421
```

Copia la carpeta `artifacts/` a tu carpeta `proyecto_crediticio/`.

---

### Paso 2 — Instalar dependencias del backend

```bash
cd proyecto_crediticio
pip install -r requirements.txt
```

> Si usas Conda: `conda activate tu_env` primero.

---

### Paso 3 — Levantar el backend

```bash
uvicorn app.main:app --reload --port 8000
```

Verás en terminal:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     ✓ Modelo XGBoost cargado  · features: ['Age', ...]
```

Verifica en el navegador: **http://127.0.0.1:8000/health**
Documentación automática: **http://127.0.0.1:8000/docs**

---

### Paso 4 — Abrir el frontend

```bash
# Opción A: servidor Python simple
python -m http.server 8080

# Luego abre: http://localhost:8080
```

O simplemente haz doble clic en `index.html` (abre como archivo local).
Asegúrate de que en `app.js` esté:

```js
const API_BASE    = "http://127.0.0.1:8000";
const USE_BACKEND = true;
```

---

## I) COMANDOS EXACTOS PARA CORRERLO LOCALMENTE

```bash
# Terminal 1 — Backend
cd proyecto_crediticio
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Terminal 2 — Frontend (opcional, también funciona con doble clic)
cd proyecto_crediticio
python -m http.server 8080
```

---

## PRUEBAS

### curl
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 38,
    "Income": 72000,
    "LoanAmount": 45000,
    "CreditScore": 680,
    "MonthsEmployed": 84,
    "NumCreditLines": 3
  }'
```

Respuesta esperada:
```json
{
  "model": "XGBoost",
  "default_probability": 0.3421,
  "predicted_class": 0,
  "risk_level": "Riesgo moderado",
  "message": "Perfil aceptable. Considerar monto menor o garantía adicional.",
  "timestamp": "2026-04-11T10:30:00.123456"
}
```

### Python (desde notebook o script)
```python
import requests

payload = {
    "Age": 38,
    "Income": 72000,
    "LoanAmount": 45000,
    "CreditScore": 680,
    "MonthsEmployed": 84,
    "NumCreditLines": 3
}

r = requests.post("http://127.0.0.1:8000/predict", json=payload)
print(r.json())
```

### Postman
- Method: POST
- URL: http://127.0.0.1:8000/predict
- Body → raw → JSON → pegar el payload de arriba
- Send

### Swagger UI (la más fácil para demo)
Abre http://127.0.0.1:8000/docs → POST /predict → "Try it out" → pegar valores → Execute

---

## J) DOCKER (OPCIONAL — solo si presentas en servidor externo)

```dockerfile
# Dockerfile (no necesario para demo local)
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Solo si tienes Docker instalado y quieres desplegar en la nube
docker build -t sistema-crediticio .
docker run -p 8000:8000 sistema-crediticio
```

**Conclusión**: para la demo de hoy, no uses Docker.

---

## FLUJO CORRECTO COMPLETO

```
1. notebook    → entrenas una vez con GridSearchCV
2. notebook    → ejecutas export_artifacts.py → genera artifacts/
3. terminal    → uvicorn app.main:app --reload
4. navegador   → abres index.html
5. sliders     → debounce 250ms → POST /predict → FastAPI → XGBoost → JSON
6. formulario  → POST /predict + análisis IA vía Anthropic API
7. historial   → registros en memoria de sesión
```
